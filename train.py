import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from datasets import RandomTransformationDataset
from models import siamese_resnet50


if __name__ == "__main__":
    dataset = RandomTransformationDataset(
        transforms=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]),
        path="/Users/ondra/dev/personal/siamese-registration/data/frame_sequences"
    )

    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    model = siamese_resnet50(1, 7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    model.to(device=device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    running_loss = 0.0
    training_loss = []

    for epoch in range(0, 10):
        with tqdm(loader, unit="batch") as progress:
            for img0, img1, params in progress:
                progress.set_description(f"Epoch {epoch}")
                img0, img1 = img0.to(device=device), img1.to(device=device)
                optimizer.zero_grad()
                outputs = model(img0, img1)
                loss = criterion(outputs, params)
                running_loss += loss
                loss.backward()
                optimizer.step()
                progress.set_postfix(loss=loss.detach())

        training_loss.append(running_loss / len(trainloader))
