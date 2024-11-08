import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image
import random
import torch.nn.functional as F
import torchvision.transforms as T
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from .utils import assert_normalized, assert_no_nans

def normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    
    # Avoid division by zero
    if max_val > min_val:
        return (tensor - min_val) / (max_val - min_val)
    else:
        return tensor - min_val  # In case max_val == min_val, return a zero tensor


def evaluate(model, dataset, device, filename, experiment=None):
    print('Start the evaluation, saving images')
    model.eval()
    dataset_length = len(dataset)
    random_indices = random.sample(range(dataset_length), 4)
    print("images to be tested: ", random_indices)
    image, mask, gt = zip(*[dataset[i] for i in random_indices])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        print("executing forward pass for evaluation...")
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    # log_var = log_var.to(torch.device('cpu'))
    
    # log_variance_clipped = torch.clamp(output[:, 1:2, :, :], min=-10, max=10)
    # print("Log_variance: min: ", log_variance_clipped.min().item(), " max:", log_variance_clipped.max().item(), " mean:", log_variance_clipped.mean().item())
    mean_output = output[:, 0:1, :, :]
    log_var = output[:, 1:2, :, :]
    
    # output_comp = mask * image + (1 - mask) * mean
    dilated_mask = dilate(mask, 25)

    # Apply Gaussian blur to both the predicted and ground truth images
    # blur_kernel_size=5
    # blur_sigma=2.0
    # gaussian_blur = T.GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)
    # mean_blurred = gaussian_blur(mean)
    # gt_blurred = gaussian_blur(gt)
    
    # normalize the variance
    normalized_variance = normalize(log_var) #replaced variance with log variance
    assert_no_nans(normalized_variance, 'normalized_variance')
    # assert_normalized(normalized_variance, 'normalized_variance')
    
    # Filter out outliers in the log_var tensor
    mean = log_var.mean()
    std = log_var.std()
    
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    
    outlier_mask = (log_var < lower_bound) | (log_var > upper_bound)
    # 4. Replace outliers with the mean value (or any other value you prefer)
    filtered_log_var = log_var.clone()
    filtered_log_var[outlier_mask] = mean
    
    # four images per row
    grid = make_grid(torch.cat([gt, mean_output, image, filtered_log_var, dilated_mask, mask], dim=0), nrow = 4)
    save_image(grid, filename)
    
    print('images saved')
    
    if experiment is not None:
        experiment.log_image(filename, filename)


def dilate(mask, kernel_size):
    # Compute the dilated binary mask
    # kernel_size = (15, 15)
    # dilated_mask = F.max_pool2d(mask, kernel_size, stride=1, padding=kernel_size[0]//2)
    # Convert tensor to numpy array
    mask_np = mask.cpu().numpy()
    # Define the structuring element
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Prepare an array to hold the dilated masks
    dilated_mask_np = np.zeros_like(mask_np)
    # Apply binary dilation to each image in the batch
    for i in range(mask_np.shape[0]):  # Loop over the batch dimension
        dilated_mask_np[i, 0] = cv2.dilate(mask_np[i, 0].astype(np.uint8), kernel)
    # Convert back to PyTorch tensor
    dilated_mask = torch.tensor(dilated_mask_np, dtype=torch.float32)
    return dilated_mask