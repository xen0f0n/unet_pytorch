from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms, utils
from PIL import Image
import glob
import matplotlib.pyplot as plt

class MembranesDataset(Dataset):
    def __init__(self, image_paths_npy, mask_paths_npy, scale=True, rotate=True, mode='train'):
        # self.image_arr = sorted(glob.glob(str(image_path) + '/*'))
        # self.target_arr = sorted(glob.glob(str(target_path) + '/*'))
        self.image_arr = np.load(image_paths_npy)
        self.mask_arr = np.load(mask_paths_npy)
        self.data_len = len(self.image_arr)
        self.mode = mode
        self.scale = scale
        self.transforms = transforms.Compose(
            [transforms.ToTensor()])

    def transform(self, image, target):
        if self.scale:
            image = TF.resize(image, 256)
            target = TF.resize(target, 256)

        params = transforms.RandomAffine.get_params(degrees=(-20, 20),
                                                    translate=(0.05, 0.05),
                                                    scale_ranges=None,
                                                    shears=(-25,25),
                                                    img_size=image.size)

        image = TF.affine(image, *params)
        target = TF.affine(target, *params)

        # plt.imshow(image)
        # plt.show()
        # plt.imshow(target)
        # plt.show()
        return image, target

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        image = Image.open(single_image_name)

        target_name = self.mask_arr[index]
        target = Image.open(target_name)

        if self.mode == 'train':
            image, target = \
                self.transform(image, target)

        img_as_np = np.asarray(image)
        img_as_np = (img_as_np - img_as_np.mean()) / img_as_np.std()
        img_as_np = np.expand_dims(img_as_np, axis=0)
        img_as_tensor = torch.from_numpy(img_as_np).float()

        target_as_np = np.asarray(target)
        target_as_np = np.expand_dims(target_as_np, axis=0)
        target_as_np = np.where(target_as_np >= 40, 1.0, 0.)
        target_as_tensor = torch.from_numpy(target_as_np).float()

        return img_as_tensor, target_as_tensor

    def __len__(self):
        return len(self.image_arr)
