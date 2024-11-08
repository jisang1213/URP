from glob import glob
from PIL import Image
import random
from torch.utils.data import Dataset
import torch
import numpy as np

from PIL import Image
import numpy as np
import torch
from PIL import Image
import numpy as np
import torch


### CHECKED ###


#Custom transform to convert image to tensor with division of pixel values by 255.0
class CustomToTensor:
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # Handle numpy array (H, W, C) -> (C, H, W)
            if pic.ndim == 3:
                img = torch.from_numpy(pic.transpose((2, 0, 1)))  # (H, W, C) -> (C, H, W)
            elif pic.ndim == 2:
                img = torch.from_numpy(pic.unsqueeze(0))  # (H, W) -> (1, H, W)
            else:
                raise ValueError("Unsupported numpy array shape. Expected 2D or 3D array.")
        elif isinstance(pic, Image.Image):
            # Handle PIL Image
            nchannel = len(pic.getbands())
            img = torch.tensor(np.array(pic), dtype=torch.uint8)  # Convert PIL image to numpy array
            img = img.permute(2, 0, 1) if nchannel == 3 else img.unsqueeze(0)  # (H, W, C) -> (C, H, W) or (1, H, W)
        else:
            raise TypeError(f'pic should be PIL Image or ndarray. Got {type(pic)}')

        return img.float() / 255.0  # Normalize to [0, 1]


class Mapdata(Dataset):
    def __init__(self, data_root, img_transform, mask_transform, data='train'):
        super(Mapdata, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # Define patterns for training or other data
        if data == 'train':
            self.heightmap_paths = sorted(glob(f'{data_root}/heightmap/heightmap*.png'))
            self.mask_paths = sorted(glob(f'{data_root}/mask/mask*.png'))
            self.true_map_paths = sorted(glob(f'{data_root}/true_map/true_map*.png'))
        else: # test set is the same as the train set for now
            self.heightmap_paths = sorted(glob(f'{data_root}/heightmap/heightmap*.png'))
            self.mask_paths = sorted(glob(f'{data_root}/mask/mask*.png'))
            self.true_map_paths = sorted(glob(f'{data_root}/true_map/true_map*.png'))

        # Ensure that the number of heightmaps matches the number of masks
        assert len(self.heightmap_paths) == len(self.mask_paths), "Mismatch between number of heightmaps and masks"
        assert len(self.heightmap_paths) == len(self.true_map_paths), "Mismatch between number of heightmaps and true maps"
        print('Number of samples: {}'.format(len(self.heightmap_paths)))

    def __len__(self):
        return len(self.heightmap_paths)

    def __getitem__(self, index):
        # Load heightmap image
        heightmap = self._load_img(self.heightmap_paths[index])
        heightmap = self.img_transform(heightmap.convert('L'))  # Convert to grayscale and transform
        # print("loading heightmap from: ", self.heightmap_paths[index])
        # Load mask image
        mask = self._load_img(self.mask_paths[index])
        mask = self.mask_transform(mask.convert('L'))  # Convert to grayscale and transform
        # Load truemap image
        truemap = self._load_img(self.true_map_paths[index])
        truemap = self.img_transform(truemap.convert('L'))  # Convert to grayscale and transform
        # Return the heightmap, mask, and the true map
        return heightmap, mask, truemap

    def _load_img(self, path):
        """
        Load an image from the given path. Handle cases where the image cannot be loaded.
        """
        try:
            img = Image.open(path)
        except:
            extension = path.split('.')[-1]
            for i in range(10):
                new_path = path.split('.')[0][:-1] + str(i) + '.' + extension
                try:
                    img = Image.open(new_path)
                    break
                except:
                    continue
        return img