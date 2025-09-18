import torch
import torch.nn as nn
from torchvision.datasets import Food101
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights, MobileNetV3
from torch.optim import AdamW, Optimizer
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
from tqdm import tqdm


def train_one_epoch(model: MobileNetV3, dataloader: DataLoader, optimizer: Optimizer, loss_fn: nn.Module):
    loss_sum = 0
    for images, targets in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        predictions = model(images)

        loss = loss_fn(predictions, targets)
        loss.backward()

        loss_sum += loss.item()
        optimizer.step()

    return loss_sum


def validate_one_epoch(model: MobileNetV3, dataloader: DataLoader, loss_fn: nn.Module):
    pass


def collate_fn(arg):
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
        collate_fn=collate_fn,
    )

    n_epochs = 1000
    for e in range(n_epochs):
        stats = train_one_epoch(model, train_dataloader, optimizer, loss_fn)


if __name__ == "__main__":
    train_model()
