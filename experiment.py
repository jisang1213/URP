from comet_ml import Experiment

import torch
import os
from torchvision import transforms

from src.dataset import Mapdata
from src.dataset import CustomToTensor
from src.model import PConvUNet
from src.mymodel import UNET, weight_init
from src.loss import InpaintingLoss, VGG16FeatureExtractor
from src.train import Trainer
from src.utils import Config, load_ckpt, create_ckpt_dir
import wandb

# Set seed for reproducibility
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

# set the config
# Get the current directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the full path to the config file
config_file_path = os.path.join(current_dir, "config.yml")

config = Config(config_file_path)
config.ckpt = create_ckpt_dir()
print("Check Point is '{}'".format(config.ckpt))

# Save a copy of the config file in the ckpt directory:
config.make_copy()

# Define the used device
device = torch.device("cuda:{}".format(config.cuda_id)
                      if torch.cuda.is_available() else "cpu")
print("CUDA is available" if torch.cuda.is_available() else "CUDA is not available")

# Define the model
print("Loading the Model...")
# model = PConvUNet(finetune=config.finetune, in_ch=1, out_ch=2,
#                   layer_size=config.layer_size)

model = UNET(in_channels=1, out_channels=2)
model.apply(weight_init)

print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")

if config.finetune:
    model.load_state_dict(torch.load(config.finetune)['model'])
    
# Move models to device
model.to(device)

# Data Transformation
img_tf = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to 256x256
            CustomToTensor()                # Convert to tensor without normalization
        ])
mask_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()           # Convert to tensor with normalization
            ])

# Define the Validation set
print("Loading the Validation Dataset...")
dataset_val = Mapdata(config.data_root,
                      img_tf,
                      mask_tf,
                      data="val")

# Set the configuration for training
if config.mode == "train":

    # Define the Places2 Dataset and Data Loader
    print("Loading the Training Dataset...")
    dataset_train = Mapdata(config.data_root,
                            img_tf,
                            mask_tf,
                            data="train")

    # Define the Loss fucntion
    criterion = InpaintingLoss(VGG16FeatureExtractor(),
                               tv_loss=config.tv_loss).to(device)
    # Define the Optimizer
    lr = config.finetune_lr if config.finetune else config.initial_lr
    if config.optim == "Adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                            model.parameters()),
                                     lr=lr,
                                     weight_decay=config.weight_decay)
    elif config.optim == "SGD":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                           model.parameters()),
                                    lr=lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)

    start_iter = 0
    if config.resume:
        print("Loading the trained params and the state of optimizer...")
        start_iter = load_ckpt(config.resume,
                               [("model", model)],
                               [("optimizer", optimizer)])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Starting from iter ", start_iter)

    trainer = Trainer(start_iter, config, device, model, dataset_train,
                      dataset_val, criterion, optimizer)
    
    for epoch in range(config.num_epochs):
        trainer.iterate(epoch)
    wandb.finish()
    
exit()
