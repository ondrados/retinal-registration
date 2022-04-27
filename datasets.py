import cv2
import os
import glob
import numpy as np
import pandas as pd
import pickle
import random
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import get_random_transformation


class RandomTransformationDataset(Dataset):
    def __init__(self, transforms=None, path=None, path_prefix=None):
        self.path = path
        self.path_prefix = path_prefix
        self.transforms = transforms
        self.dataframe = pd.read_pickle(self.path)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        if self.path_prefix:
            image = cv2.imread(
                os.path.join(self.path_prefix, self.dataframe.iloc[index].relative_path),
                cv2.IMREAD_GRAYSCALE
            )
        else:
            image = cv2.imread(
                self.dataframe.iloc[index].relative_path,
                cv2.IMREAD_GRAYSCALE
            )
        trans_image, trans_image_crop, image_crop, params = get_random_transformation(image)
        if self.transforms:
            image_crop = self.transforms(image_crop)
            trans_image_crop = self.transforms(trans_image_crop)
        return image_crop, trans_image_crop, torch.FloatTensor(params)


if __name__ == '__main__':
    dataset = RandomTransformationDataset(
        transforms=transforms.Compose([
            transforms.ToTensor(),
        ]),
        path="/Users/ondra/dev/personal/retinal-registration/data/train.pkl"
    )

    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    image_crop, trans_image_crop, params = next(iter(loader))

    print(image_crop.shape)
    print(params)

    # plt.figure(figsize=(16, 9))
    # plt.subplot(121)
    # plt.imshow(np.transpose(image_crop[0, :, :, :], (1, 2, 0)))
    # plt.title('Original image')
    #
    # plt.subplot(122)
    # plt.imshow(np.transpose(trans_image_crop[0, :, :, :], (1, 2, 0)))
    # plt.title('Transformed image')
    #
    # plt.show()
