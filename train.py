import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets import RandomTransformationDataset
from models import *


parser = argparse.ArgumentParser()
parser.add_argument("--data")
parser.add_argument("--outputs")


if __name__ == "__main__":

    args = parser.parse_args()
    path = args.data
    output_path = args.outputs

    train_dataset = RandomTransformationDataset(
        transforms=transforms.Compose([
            transforms.ToTensor(),
        ]),
        path=os.path.join(path, "train.pkl")
    )

    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=False)

    test_dataset = RandomTransformationDataset(
        transforms=transforms.Compose([
            transforms.ToTensor(),
        ]),
        path=os.path.join(path, "test.pkl")
    )

    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

    pretrained = True
    model = siamese_resnet18(1, 7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    if pretrained:
        checkpoint_path = "07_resnet18_MSE_corr/checkpoint-12.pt"
        checkpoint = torch.load(os.path.join("/content/drive/MyDrive/outputs", checkpoint_path), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_check = checkpoint['epoch']
        training_loss_check = checkpoint['training_loss']
        validation_loss_check = checkpoint['validation_loss']

    print(f"Running on {device}")

    training_loss = training_loss_check if pretrained else []
    validation_loss = validation_loss_check if pretrained else []
    start = epoch_check + 1 if pretrained else 0
    end = epoch_check + 25 if pretrained else 25

    for epoch in range(start, end):
        model.train()
        running_loss = 0.0
        with tqdm(train_loader, unit="batch") as progress:
            for img0, img1, params in progress:
                progress.set_description(f"Epoch {epoch} - train")
                img0, img1, params = img0.to(device=device), img1.to(device=device), params.to(device=device)
                optimizer.zero_grad()
                outputs = model(img0, img1)
                loss = criterion(outputs, params)
                loss_item = loss.item()
                running_loss += loss_item
                loss.backward()
                optimizer.step()
                progress.set_postfix(loss=loss_item)

        training_loss.append(running_loss / len(train_loader))

        model.eval()
        val_running_loss = 0.0
        with tqdm(test_loader, unit="batch") as validation_progress:
            for img0, img1, params in validation_progress:
                validation_progress.set_description(f"Epoch {epoch} - validation")
                img0, img1, params = img0.to(device=device), img1.to(device=device), params.to(device=device)
                outputs = model(img0, img1)
                loss = criterion(outputs, params)
                loss_item = loss.item()
                val_running_loss += loss_item
                validation_progress.set_postfix(loss=loss_item)

        validation_loss.append(val_running_loss / len(test_loader))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_loss': training_loss,
            'validation_loss': validation_loss
        }, os.path.join(output_path, f"checkpoint-{epoch}.pt"))

        plt.figure()
        plt.plot(training_loss, label="training loss")
        plt.plot(validation_loss, label="validation loss")
        plt.title("Training loss")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(os.path.join(output_path, "loss.png"))
        plt.close()
