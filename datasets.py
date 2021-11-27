import cv2
import os
import glob
import numpy as np
import random
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import get_random_transformation

dataset_path = "/Users/ondra/dev/personal/siamese-registration/data/frame_sequences"


class RandomTransformationDataset(Dataset):
    def __init__(self, transforms=None, path=dataset_path):
        self.path = path

        self.transforms = transforms

        self.folder_paths = glob.glob(os.path.join(self.path, '*'))
        self.id_list = []
        self.image_list = []

        for folder in self.folder_paths:
            images = glob.glob(folder + '/*jpg')
            self.id_list.append(os.path.basename(folder))
            self.image_list.extend(images)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index])
        trans_image, trans_image_crop, image_crop, params = get_random_transformation(image)
        if self.transforms:
            image_crop = self.transforms(image_crop)
            trans_image_crop = self.transforms(trans_image_crop)
        return image_crop, trans_image_crop, torch.FloatTensor(params)


if __name__ == '__main__':
    dataset = RandomTransformationDataset(
        transforms=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]),
        path="/Users/ondra/dev/personal/siamese-registration/data/frame_sequences"
    )

    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
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
