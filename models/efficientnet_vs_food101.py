import torch
import torch.nn as nn
from torchvision.datasets import Food101
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights, EfficientNet
from torch.optim import AdamW, Optimizer
from torch.utils.data.dataloader import DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import torchvision.transforms as transforms


class Settings:
    DEVICE = "cuda"
    BATCH_SIZE = 32
    MODEL_NAME = "efficientnet"
    OUT_DIR = Path(__file__).parent / "efficientnet_train"


def train_one_epoch(model: EfficientNet, dataloader: DataLoader, optimizer: Optimizer, loss_fn: nn.Module):
    loss_sum = 0
    for images, targets in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        images = images.to(Settings.DEVICE)
        targets = targets.to(Settings.DEVICE)

        predictions = model(images)

        loss = loss_fn(predictions, targets)
        loss.backward()

        loss_sum += loss.item()
        optimizer.step()

    return {
        "loss_sum": loss_sum
    }


def batch_confusion(predictions: torch.Tensor, targets: torch.Tensor):
    batch_confusion = np.zeros((101, 101))
    for actual, predicted in zip(targets, predictions.argmax(1)):
        batch_confusion[actual, predicted] += 1

    return batch_confusion


def validate_one_epoch(model: EfficientNet, dataloader: DataLoader, loss_fn: nn.Module):
    confusion_matrix = np.zeros((101, 101))
    loss_sum = 0
    for images, targets in tqdm(dataloader, desc="validating", leave=False):
        images = images.to(Settings.DEVICE)
        targets = targets.to(Settings.DEVICE)

        predictions = model(images)

        loss = loss_fn(predictions, targets)

        confusion_matrix += batch_confusion(predictions, targets)

        loss_sum += loss.item()

    return {
        "confusion_matrix": confusion_matrix,
        "loss_sum": loss_sum
    }


def get_collate_fn(train):
    def anonymous(arg):
        if train:
            return pil_to_tensor_and_stuff(arg, apply_augmentation=True)
        else:
            return pil_to_tensor_and_stuff(arg, apply_augmentation=False)

    return anonymous


def pil_to_tensor_and_stuff(arg, apply_augmentation):
    targets = []
    tensor_images = []

    my_transforms = [
        transforms.Resize((480, 480)),
    ]

    if apply_augmentation:
        my_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
        my_transforms.append(transforms.RandomPerspective(
            distortion_scale=0.2, p=0.5))
        my_transforms.append(transforms.RandomRotation(degrees=15))

    my_transforms += [
        transforms.ToTensor(),
        # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_l.html#torchvision.models.EfficientNet_V2_L_Weights
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]  # a bit weird values?
        )
    ]

    my_transforms = transforms.Compose(my_transforms)

    for img, clazz in arg:
        targets.append(clazz)

        img_tensor = my_transforms(img)

        tensor_images.append(img_tensor)

    # convert to batch tensors (B, class) and (B, C, H, W)
    targets = torch.tensor(targets)
    tensor_images = torch.stack(tensor_images)

    # print(tensor_images.shape)
    return tensor_images, targets


def train_model():
    train = Food101(root="./", split="train", download=True)
    val = Food101(root="./", split="test", download=True)

    os.makedirs(Settings.OUT_DIR)

    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    model.classifier = nn.Sequential(
        nn.Dropout(0.2, inplace=True),
        nn.Linear(1280, 101),
    )
    model = model.to(Settings.DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=0.0005)
    train_dataloader = DataLoader(
        dataset=train,
        shuffle=True,
        batch_size=Settings.BATCH_SIZE,
        collate_fn=get_collate_fn(train=True),
        persistent_workers=True,
        num_workers=4
    )

    val_dataloader = DataLoader(
        dataset=val,
        batch_size=Settings.BATCH_SIZE,
        collate_fn=get_collate_fn(train=False),
        persistent_workers=True,
        num_workers=4
    )

    n_epochs = 100

    val_accuracies = []
    val_losses = []
    train_losses = []
    lowest_val_loss = float('inf')
    highest_accuracy = 0
    for e in range(n_epochs):
        print("*" * 10, "Epoch", e+1, "*" * 10)
        model.train()
        train_stats = train_one_epoch(
            model, train_dataloader, optimizer, loss_fn)
        print("train loss:", train_stats["loss_sum"])
        train_losses.append(train_stats["loss_sum"])

        model.eval()
        with torch.no_grad():
            val_stats = validate_one_epoch(model, val_dataloader, loss_fn)

            print("val loss:", val_stats["loss_sum"])
            confusion_matrix = val_stats["confusion_matrix"]
            num_correct = confusion_matrix.trace()
            total = confusion_matrix.sum()
            val_accuracy = num_correct / total
            print("val acc:", val_accuracy)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_stats["loss_sum"])

        plt.title("Loss")
        plt.plot(val_losses, label="validation")
        plt.plot(train_losses, label="training")
        plt.legend()
        plt.savefig(Settings.OUT_DIR / "loss.png")
        plt.close()

        plt.title("Accuracy")
        plt.plot(val_accuracies, label="validation")
        plt.legend()
        plt.savefig(Settings.OUT_DIR / "validation.png")
        plt.close()

        if val_stats["loss_sum"] < lowest_val_loss:
            lowest_val_loss = val_stats["loss_sum"]
            torch.jit.script(model).save(Settings.OUT_DIR / "model_loss.pt")
            torch.save(model.state_dict(), Settings.OUT_DIR /
                       "model_loss_state_dict.pt")

        if val_accuracy > highest_accuracy:
            highest_accuracy = val_accuracy
            torch.jit.script(model).save(
                Settings.OUT_DIR / "model_accuracy.pt")
            torch.save(model.state_dict(), Settings.OUT_DIR /
                       "model_state_dict_accuracy.pt")

        with open(Settings.OUT_DIR / "stats.csv", "+a") as f:
            if e == 0:
                f.write("train loss,val loss, val acc\n")

            f.write(
                f"{str(train_stats['loss_sum'])}, {str(val_stats['loss_sum'])}, {str(val_accuracy)}\n")


if __name__ == "__main__":
    train_model()
