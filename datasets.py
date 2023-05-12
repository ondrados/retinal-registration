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

from utils import get_random_transformation, get_random_translation


class RandomTransformationDataset(Dataset):
    def __init__(self, transforms=None, path=None, path_prefix=None, tr_only=False):
        self.path = path
        self.path_prefix = path_prefix
        self.transforms = transforms
        self.dataframe = pd.read_pickle(self.path)
        self.tr_only = tr_only

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
        if self.tr_only:
            trans_image, trans_image_crop, image_crop, params = get_random_translation(image)
        else:
            trans_image, trans_image_crop, image_crop, params = get_random_transformation(image)
        if self.transforms:
            image_crop = self.transforms(image_crop)
            trans_image_crop = self.transforms(trans_image_crop)
        return image_crop, trans_image_crop, torch.FloatTensor(params)


class FolderFramesDataset(Dataset):
    def __init__(self, transforms=None, path=None):
        self.path = path
        self.transforms = transforms
        self.folder_paths = glob.glob(os.path.join(self.path, '*'))
        self.id_list = []
        self.image_list = []

        for folder in self.folder_paths:
            images = glob.glob(folder + '/*jpg')
            self.id_list.append(os.path.basename(folder))
            self.image_list.extend(images)
            break

    def __len__(self):
        return len(self.image_list) - 1

    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index], cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(self.image_list[index+1], cv2.IMREAD_GRAYSCALE)

        h = 500
        w = int(h * 1.5)
        center = (image.shape[1] // 2, image.shape[0] // 2)

        image_crop = image[center[1] - h // 2:center[1] + h // 2, center[0] - w // 2:center[0] + w // 2]
        image2_crop = image2[center[1] - h // 2:center[1] + h // 2, center[0] - w // 2:center[0] + w // 2]

        if self.transforms:
            image_crop = self.transforms(image_crop)
            image2_crop = self.transforms(image2_crop)

        return image_crop, image2_crop


if __name__ == '__main__':
    dataset = RandomTransformationDataset(
        transforms=transforms.Compose([
            transforms.ToTensor(),
        ]),
        path="/Users/ondrados/dev/school/retinal-registration/data/train.pkl",
        tr_only=True
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
