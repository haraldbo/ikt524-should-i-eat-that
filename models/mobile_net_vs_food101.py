import torch
import torch.nn as nn
from torchvision.datasets import Food101
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights, MobileNetV3
from torch.optim import AdamW, Optimizer
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
from tqdm import tqdm
import numpy as np


def train_one_epoch(model: MobileNetV3, dataloader: DataLoader, optimizer: Optimizer, loss_fn: nn.Module):
    loss_sum = 0
    for images, targets in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
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


def validate_one_epoch(model: MobileNetV3, dataloader: DataLoader, loss_fn: nn.Module):
    confusion_matrix = np.zeros((101, 101))
    loss_sum = 0
    for images, targets in tqdm(dataloader, desc="validating"):
        predictions = model(images)
        loss = loss_fn(predictions, targets)

        confusion_matrix += batch_confusion(predictions, targets)

        loss_sum += loss.item()

    return {
        "confusion_matrix": confusion_matrix,
        "loss_sum": loss_sum
    }


def pil_to_tensor_and_stuff(arg):
    targets = []
    tensor_images = []
    for img, clazz in arg:
        targets.append(clazz)

        # resize images
        img: Image.Image
        img = img.resize((224, 224))

        # convert to tensor and normalize it
        img_tensor = pil_to_tensor(img).type(torch.float32)
        img_tensor -= 256/2
        img_tensor /= 256

        tensor_images.append(img_tensor)

    # convert to batch tensors (B, class) and (B, C, H, W)
    targets = torch.tensor(targets)
    tensor_images = torch.stack(tensor_images)

    # print(tensor_images.shape)
    return tensor_images, targets


def train_model():
    train = Food101(root="./", split="train", download=True)
    val = Food101(root="./", split="test", download=True)
    model = mobilenet_v3_large(
        weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)

    model.features
    model.classifier = nn.Sequential(
        nn.Linear(960, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 101)
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=0.001)
    train_dataloader = DataLoader(
        dataset=train,
        shuffle=True,
        batch_size=32,
        collate_fn=pil_to_tensor_and_stuff,
    )

    val_dataloader = DataLoader(
        dataset=val,
        batch_size=32,
        collate_fn=pil_to_tensor_and_stuff
    )

    n_epochs = 1000
    for e in range(n_epochs):
        model.train()
#        stats = train_one_epoch(model, train_dataloader, optimizer, loss_fn)

        model.eval()
        with torch.no_grad():
            stats = validate_one_epoch(model, val_dataloader, loss_fn)

            confusion_matrix = stats["confusion_matrix"]
            


if __name__ == "__main__":
    train_model()
