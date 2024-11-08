import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torchvision.transforms as T
from .utils import assert_no_nans
import pytorch_msssim

class InpaintingLoss(nn.Module):
    def __init__(self, extractor, tv_loss='mean'):
        super(InpaintingLoss, self).__init__()
        self.tv_loss = tv_loss
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        # default extractor is VGG16
        self.extractor = extractor

    def forward(self, input, mask, output, gt):

        # Compute the dilated binary mask
        kernel_size = 25
        dilated_mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        assert_no_nans(dilated_mask, 'dilated_mask')

        #seperate the output image channel and the variance channel here
        mean = output[:, 0:1, :, :] # the first channel is the output
        log_variance = output[:, 1:2, :, :] #the second channel is the variance
        
        # Uncertainty loss over the dilated region
        variance_loss = uncertainty_loss(mean, log_variance, gt, dilated_mask)
        assert_no_nans(variance_loss, "variance loss")
        
        # Regularization term to penalize large log_variance
        log_var_L2reg = torch.mean((dilated_mask*log_variance) ** 2)
        
        # Non-hole pixels directly set to ground truth
        comp = mask * input + (1 - mask) * mean
        # Total Variation Regularization
        tv_loss = total_variation_loss(comp, mask, self.tv_loss)
        # tv_loss = (torch.mean(torch.abs(comp[:, :, :, :-1] - comp[:, :, :, 1:])) \
        #           + torch.mean(torch.abs(comp[:, :, :, 1:] - comp[:, :, :, :-1])) \
        #           + torch.mean(torch.abs(comp[:, :, :-1, :] - comp[:, :, 1:, :])) \
        #           + torch.mean(torch.abs(comp[:, :, 1:, :] - comp[:, :, :-1, :]))) / 2
        
        # Thresholded TV (loss only computed for differences less than threshold)
        thresholded_tv = thresholded_total_variation_loss(mean, threshold=0.03)
        
        # Reconstructed Pixel Loss over Pixels that are inside dilated mask but outside mask:
        reconstruct_mask = dilated_mask - mask
        reconstruction_loss = self.l1(reconstruct_mask * mean, reconstruct_mask * gt)
        # Valid Pixel Loss
        valid_loss = self.l1(mask * mean, mask * gt)
        
        # MSE Loss over dilated region
        dilated_MSE_loss = self.mse(dilated_mask * mean, dilated_mask * gt)

        # Sobel loss (edge)
        edge_loss = sobel_loss(mean, gt, dilated_mask, blur_kernel_size=7, blur_sigma=3.0)
        
        ssim_loss = ms_ssim_loss(mean, gt)

        # Perceptual Loss and Style Loss
        feats_out = self.extractor(mean)
        feats_comp = self.extractor(comp)
        feats_gt = self.extractor(gt)
        perc_loss = 0.0
        style_loss = 0.0
        # Calculate the L1Loss for each feature map
        for i in range(3):
            perc_loss += self.l1(feats_out[i], feats_gt[i])
            perc_loss += self.l1(feats_comp[i], feats_gt[i])
            style_loss += self.l1(gram_matrix(feats_out[i]),
                                  gram_matrix(feats_gt[i]))
            style_loss += self.l1(gram_matrix(feats_comp[i]),
                                  gram_matrix(feats_gt[i]))

        return {'valid': valid_loss,
                'reconstruction': reconstruction_loss,
                'MSE' : dilated_MSE_loss,
                'edge' : edge_loss,
                'perc': perc_loss,
                'style': style_loss,
                'tv': tv_loss,
                'thresholded_tv' : thresholded_tv,
                'variance': variance_loss,
                'log_var_L2reg' : log_var_L2reg,
                'ssim' : ssim_loss}


# The network of extracting the feature for perceptual and style loss
class VGG16FeatureExtractor(nn.Module):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(weights = models.VGG16_Weights.DEFAULT)
        normalization = Normalization(self.MEAN, self.STD)
        # Define the each feature exractor
        self.enc_1 = nn.Sequential(normalization, *vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{}'.format(i+1)).parameters():
                param.requires_grad = False

    def forward(self, input):
        feature_maps = [input]
        for i in range(3):
            feature_map = getattr(self, 'enc_{}'.format(i+1))(feature_maps[-1])
            feature_maps.append(feature_map)
        return feature_maps[1:]


# Normalization Layer for VGG
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, input):
        # normalize img
        if self.mean.type() != input.type():
            self.mean = self.mean.to(input)
            self.std = self.std.to(input)
        return (input - self.mean) / self.std


# Calcurate the Gram Matrix of feature maps
def gram_matrix(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def dilation_holes(hole_mask):
    # b, ch, h, w = hole_mask.shape
    # dilation_conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False).to(hole_mask)
    # torch.nn.init.constant_(dilation_conv.weight, 1.0)
    # with torch.no_grad():
    #     output_mask = dilation_conv(hole_mask)
    # updated_holes = output_mask != 0
    # return updated_holes.float()
    kernel_size = 5
    dilated_mask = F.max_pool2d(hole_mask, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    return dilated_mask


def thresholded_total_variation_loss(x, threshold=0.03):
    # Calculate the difference between adjacent pixels in both dimensions
    diff_i = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    diff_j = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    
    # Apply the threshold: Only penalize differences smaller than the threshold
    tv_loss_i = torch.mean(torch.where(diff_i < threshold, diff_i, torch.zeros_like(diff_i)))
    tv_loss_j = torch.mean(torch.where(diff_j < threshold, diff_j, torch.zeros_like(diff_j)))
    
    return tv_loss_i + tv_loss_j

def total_variation_loss(image, mask, method):
    hole_mask = 1 - mask
    dilated_holes = dilation_holes(hole_mask)
    colomns_in_Pset = dilated_holes[:, :, :, 1:] * dilated_holes[:, :, :, :-1]
    rows_in_Pset = dilated_holes[:, :, 1:, :] * dilated_holes[:, :, :-1:, :]
    if method == 'sum':
        loss = torch.sum(torch.abs(colomns_in_Pset*(
                    image[:, :, :, 1:] - image[:, :, :, :-1]))) + \
            torch.sum(torch.abs(rows_in_Pset*(
                    image[:, :, :1, :] - image[:, :, -1:, :])))
    else:
        loss = torch.mean(torch.abs(colomns_in_Pset*(
                    image[:, :, :, 1:] - image[:, :, :, :-1]))) + \
            torch.mean(torch.abs(rows_in_Pset*(
                    image[:, :, :1, :] - image[:, :, -1:, :])))
    return loss

def uncertainty_loss(mean, log_variance, gt, dilated_mask):
    # Compute the exponential of the log variances to get variances
    
    log_variance = torch.clamp(log_variance, -10, 10)
    
    mse_term = 0.5 * torch.exp(-log_variance) * (mean - gt) ** 2
    log_term = 0.5 * log_variance
    
    # Uncertainty Loss - add variance_reg if unstable, add -log_variance to enforce positive value
    uncertainty_loss = torch.mean(dilated_mask*(mse_term + log_term))

    return uncertainty_loss

def sobel_loss(mean, gt, dilated_mask, blur_kernel_size=5, blur_sigma=2.0):
    """Computes the Sobel loss with Gaussian blur applied before Sobel operator."""
    
    # Define Sobel kernels
    sobel_kernel_x = torch.tensor([[-1., 0., 1.], 
                                   [-2., 0., 2.], 
                                   [-1., 0., 1.]]).view(1, 1, 3, 3)

    sobel_kernel_y = torch.tensor([[-1., -2., -1.], 
                                   [ 0.,  0.,  0.], 
                                   [ 1.,  2.,  1.]]).view(1, 1, 3, 3)
    
    device = mean.device
    sobel_kernel_x = sobel_kernel_x.to(device)
    sobel_kernel_y = sobel_kernel_y.to(device)
    
    # Apply Gaussian blur to both the predicted and ground truth images
    gaussian_blur = T.GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)
    mean_blurred = gaussian_blur(mean)
    gt_blurred = gaussian_blur(gt)
    
    # Calculate gradients by applying the Sobel filter
    Gx_pred = F.conv2d(mean_blurred, sobel_kernel_x, padding=1)
    Gy_pred = F.conv2d(mean_blurred, sobel_kernel_y, padding=1)
    Gx_gt = F.conv2d(gt_blurred, sobel_kernel_x, padding=1)
    Gy_gt = F.conv2d(gt_blurred, sobel_kernel_y, padding=1)

    # Compute the gradient differences
    grad_diff_x = Gx_gt - Gx_pred
    grad_diff_y = Gy_gt - Gy_pred

    # Compute Sobel loss as the mean squared error of the gradient differences
    sobel_loss = torch.mean(dilated_mask * (grad_diff_x ** 2 + grad_diff_y ** 2))

    return sobel_loss

def ms_ssim_loss(img1, img2):
    return 1 - pytorch_msssim.ms_ssim(img1, img2, data_range=1.0, size_average=True)


if __name__ == '__main__':
    from config import get_config
    config = get_config()
    vgg = VGG16FeatureExtractor()
    criterion = InpaintingLoss(config['loss_coef'], vgg)

    img = torch.randn(1, 1, 256, 256)
    mask = torch.ones((1, 1, 256, 256))
    mask[:, :, 120:, :][:, :, :, 120:] = 0
    input = img * mask
    out = torch.randn(1, 1, 125, 125)
    loss = criterion(input, mask, out, img)
    
    







# def sobel_loss(mean, gt, dilated_mask):
#     # Define Sobel kernels
#     sobel_kernel_x = torch.tensor([[-1., 0., 1.], 
#                                 [-2., 0., 2.], 
#                                 [-1., 0., 1.]]).view(1, 1, 3, 3)

#     sobel_kernel_y = torch.tensor([[-1., -2., -1.], 
#                                 [ 0.,  0.,  0.], 
#                                 [ 1.,  2.,  1.]]).view(1, 1, 3, 3)
    
#     device = mean.device
#     sobel_kernel_x = sobel_kernel_x.to(device)
#     sobel_kernel_y = sobel_kernel_y.to(device)
    
    
    
#     # Calculate gradients by applying sobel filter
#     Gx_pred = F.conv2d(mean, sobel_kernel_x, padding=1)
#     Gy_pred = F.conv2d(mean, sobel_kernel_y, padding=1)
#     Gx_gt = F.conv2d(gt, sobel_kernel_x, padding=1)
#     Gy_gt = F.conv2d(gt, sobel_kernel_y, padding=1)

#     # Compute the gradient differences
#     grad_diff_x = Gx_gt - Gx_pred
#     grad_diff_y = Gy_gt - Gy_pred

#     # Compute Sobel loss as the mean squared error of the gradient differences
#     sobel_loss = torch.mean(dilated_mask*(grad_diff_x**2 + grad_diff_y**2))

#     # # Compute magnitude of gradients (this explodes)
#     # magnitude_pred = torch.sqrt(Gx_pred ** 2 + Gy_pred ** 2)
#     # magnitude_gt = torch.sqrt(Gx_gt ** 2 + Gy_gt ** 2)

#     # # Compute Sobel loss over the dilated_mask (L1 Loss in this example)
#     # sobel_loss = torch.mean(dilated_mask * torch.abs(magnitude_pred - magnitude_gt))

#     # Alternatively, you can use L2 Loss
#     # sobel_loss = torch.mean((magnitude_pred - magnitude_gt) ** 2)
#     return sobel_loss
