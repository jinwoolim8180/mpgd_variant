import torch
import os
from scipy import io
import glob
import cv2
import numpy as np
from PIL import Image

def my_zero_pad(img, block_size=32):
    old_h, old_w, _ = img.shape
    delta_h = (block_size - np.mod(old_h, block_size)) % block_size
    delta_w = (block_size - np.mod(old_w, block_size)) % block_size
    img_pad = np.concatenate((img, np.zeros([old_h, delta_w, 3])), axis=1)
    img_pad = np.concatenate((img_pad, np.zeros([delta_h, old_w + delta_w, 3])), axis=0)
    new_h, new_w, _ = img_pad.shape
    return img, old_h, old_w, img_pad, new_h, new_w

class Dataset():
    def __init__(self,root='dataset', train=True, transform=None,target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.test_image_paths = glob.glob(os.path.join(root, "*"))

        self.data = []

        # for i, path in enumerate(test_image_paths):
        #     test_image = cv2.imread(path, 1)  # read test data from image file
        #     test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        #     # img, old_h, old_w, img_pad, new_h, new_w = my_zero_pad(test_image, block_size=32)
        #     # img_pad = img_pad.reshape(new_h, new_w, 3) / 255.0  # normalization
        #     self.data.append(test_image)


    def __len__(self):
        return len(self.test_image_paths)

    def __getitem__(self, index):
        test_image = cv2.imread(self.test_image_paths[index], 1)  # read test data from image file
        img = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        
        img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, 0