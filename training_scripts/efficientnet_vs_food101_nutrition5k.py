import torch
import torch.nn as nn
from torchvision.datasets import Food101
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights, EfficientNet
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights, MobileNetV3
from torch.optim import AdamW, Optimizer
from torch.utils.data.dataloader import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import torchvision.transforms as transforms
import random


class Settings:
    DEVICE = "cuda"
    BATCH_SIZE = 16
    MODEL_NAME = "efficientnet"
    OUT_DIR = Path(__file__).parent / "efficientnet_train"


def train_one_epoch(model: EfficientNet, dataloader: DataLoader, optimizer: Optimizer, loss_fn: nn.Module):
    nutri_loss_sum = 0.0
    class_loss_sum = 0.0
    for batch_tensor_images, batch_target_classes, batch_nutrientz, batch_sample_has_nutriz in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()

        images = batch_tensor_images.to(Settings.DEVICE)
        targets = batch_target_classes.to(Settings.DEVICE)
        batch_nutrientz = batch_nutrientz.to(Settings.DEVICE)
        batch_sample_has_nutriz = batch_sample_has_nutriz.to(Settings.DEVICE)

        out = model(images)
        predicted_class = out[:, :101]
        predicted_nutrients = out[:, 101:]

        class_pred_loss = loss_fn(predicted_class, targets)
        if not torch.isnan(class_pred_loss):
            class_pred_loss.backward(retain_graph=True)
            class_loss_sum += class_pred_loss.item()

        nutri_loss = (((batch_nutrientz - predicted_nutrients) ** 2).sum(dim=1)
                      * batch_sample_has_nutriz).sum(0)/batch_sample_has_nutriz.sum(0)
        if not torch.isnan(nutri_loss):
            nutri_loss.backward()
            nutri_loss_sum += nutri_loss.item()

        optimizer.step()

    return {
        "loss_sum": nutri_loss_sum + class_loss_sum,
        "nutri_loss_sum": nutri_loss_sum,
        "class_loss_sum": class_loss_sum,
    }


def batch_confusion(predictions: torch.Tensor, targets: torch.Tensor):
    batch_confusion = np.zeros((101, 101))
    for actual, predicted in zip(targets, predictions.argmax(1)):
        batch_confusion[actual, predicted] += 1

    return batch_confusion


def validate_one_epoch(model: EfficientNet, dataloader: DataLoader, loss_fn: nn.Module):
    confusion_matrix = np.zeros((101, 101))
    nutri_loss_sum = 0.0
    class_loss_sum = 0.0
    for batch_tensor_images, batch_target_classes, batch_nutrientz, batch_sample_has_nutriz in tqdm(dataloader, desc="validating", leave=False):

        images = batch_tensor_images.to(Settings.DEVICE)
        targets = batch_target_classes.to(Settings.DEVICE)
        batch_nutrientz = batch_nutrientz.to(Settings.DEVICE)
        batch_sample_has_nutriz = batch_sample_has_nutriz.to(Settings.DEVICE)
        out = model(images)
        predicted_class = out[:, :101]
        predicted_nutrients = out[:, 101:]
        confusion_matrix += batch_confusion(predicted_class, targets)

        class_pred_loss = loss_fn(predicted_class, targets)
        if not torch.isnan(class_pred_loss):
            class_loss_sum += class_pred_loss.item()

        nutri_loss = (((batch_nutrientz - predicted_nutrients) ** 2).sum(dim=1)
                      * batch_sample_has_nutriz).sum(0)/batch_sample_has_nutriz.sum(0)
        if not torch.isnan(nutri_loss):
            nutri_loss_sum += nutri_loss.item()

    return {
        "loss_sum": nutri_loss_sum + class_loss_sum,
        "nutri_loss_sum": nutri_loss_sum,
        "class_loss_sum": class_loss_sum,
        "confusion_matrix": confusion_matrix
    }


def get_collate_fn(train):
    def anonymous(arg):
        if train:
            return pil_to_tensor_and_stuff(arg, apply_augmentation=True)
        else:
            return pil_to_tensor_and_stuff(arg, apply_augmentation=False)

    return anonymous


def pil_to_tensor_and_stuff(arg, apply_augmentation):
    batch_target_classes = []
    batch_tensor_images = []
    batch_nutrientz = []
    batch_has_nutrients = []

    my_transforms = [
        transforms.Resize((480, 480)),
    ]

    if apply_augmentation:
        my_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
        my_transforms.append(transforms.RandomPerspective(
            distortion_scale=0.2, p=0.5))
        my_transforms.append(transforms.RandomRotation(degrees=20))

    my_transforms += [
        transforms.ToTensor(),
        # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_l.html#torchvision.models.EfficientNet_V2_L_Weights
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ]

    my_transforms = transforms.Compose(my_transforms)

    for img, target_class, nutrients, has_nutriets in arg:
        batch_target_classes.append(target_class)
        batch_nutrientz.append(torch.tensor(nutrients, dtype=torch.float32))
        batch_has_nutrients.append(has_nutriets)
        img_tensor = my_transforms(img)

        batch_tensor_images.append(img_tensor)

    # convert to batch tensors (B, class) and (B, C, H, W)

    # scale and normalize nutrisents
    batch_nutrientz = torch.stack(batch_nutrientz)
    # means = [195.3, 12.5, 16.2, 9.8]
    nutri_means = torch.tensor([195.3, 12.5, 16.2, 9.8]).reshape((1, 4))
    # std = [313.1, 16.7, 29.8, 25.5]
    nutri_stds = torch.tensor([313.1, 16.7, 29.8, 25.5]).reshape((1, 4))

    batch_nutrientz = (batch_nutrientz - nutri_means)/nutri_stds

    batch_target_classes = torch.tensor(batch_target_classes)
    batch_tensor_images = torch.stack(batch_tensor_images)
    batch_has_nutrients = torch.tensor(batch_has_nutrients, dtype=torch.int)

    # print(tensor_images.shape)
    # img, target_class, calories, total_fat, total_carb, total_protein, sample_origin
    return batch_tensor_images, batch_target_classes, batch_nutrientz, batch_has_nutrients


def get_nutrition5k_datasets(val_ratio=0.2):
    i5k_meta = _load_meta_data()
    n = int(len(i5k_meta) * val_ratio)

    i5k_test = i5k_meta[:n]
    i5k_train = i5k_meta[n:]

    return Nutrition5k(i5k_train, train=True), Nutrition5k(i5k_test, train=False)


def _load_meta_data():
    dishes = os.listdir(Path("n5k"))
    carbs = []
    proteins = []
    fats = []
    calories = []

    with open("dataset_analysis/nutrition5k/cleaned_dish_metadata.csv") as f:
        # ignore csv header
        # dish_id,total_calories,total_mass,total_fat,total_carb,total_protein,num_ingrs
        f.readline()
        data = []
        count = 0
        for line in f.readlines():
            line = line.strip()
            dish_id, dish_calories, dish_mass, dish_fat, dish_carb, dish_protein, _ = line.split(
                ",")
            if dish_id not in dishes:
                # print(dish_id, "not found")
                count += 1
                continue

            dish_calories = float(dish_calories)
            dish_mass = float(dish_mass)
            dish_fat = float(dish_fat)
            dish_carb = float(dish_carb)
            dish_protein = float(dish_protein)

            carbs.append(dish_carb)
            proteins.append(dish_protein)
            fats.append(dish_fat)
            calories.append(dish_calories)

            data.append({
                "dish_id": dish_id,
                "total_calories": dish_calories,
                "total_mass": dish_mass,
                "total_fat": dish_fat,
                "total_carb": dish_carb,
                "total_protein": dish_protein,
            })

        print("Total number of dishes not found:", count)
        print("*" * 15, "Dish stats", "*" * 15)
        print("Mean:")
        print("\tcalories:", np.mean(calories))
        print("\tprotein:", np.mean(proteins))
        print("\tcarbs:", np.mean(carbs))
        print("\tfats:", np.mean(fats))
        print("Std:")
        print("\tcalories:", np.std(calories))
        print("\tprotein:", np.std(proteins))
        print("\tcarbs:", np.std(carbs))
        print("\tfats:", np.std(fats))

        # calories, protein, carbs, fats
        # means = [195.3, 12.5, 16.2, 9.8]
        # std = [313.1, 16.7, 29.8, 25.5]

        return data


class Nutrition5k(Dataset):

    def __init__(self, meta_csv, train):
        self.train = train
        self.meta_csv = meta_csv
        # works for now but prolaby good to have this configurable
        self.root_dir = Path("n5k")

    def __len__(self):
        return len(self.meta_csv)

    def __getitem__(self, index):

        dish_id = self.meta_csv[index]["dish_id"]
        total_calories = self.meta_csv[index]["total_calories"]
        total_fat = self.meta_csv[index]["total_fat"]
        total_carb = self.meta_csv[index]["total_carb"]
        total_protein = self.meta_csv[index]["total_protein"]

        images_dir = self.root_dir / dish_id / "frames_sampled30"
        if self.train:
            # pick any one of the sampled images
            img_filename = random.choice(os.listdir(images_dir))

        else:
            # pick the first one
            img_filename = os.listdir(images_dir)[0]

        img = Image.open(images_dir / img_filename)
        # the images are upside down for some reason. rotating 180 degs them to crrect it
        img = img.rotate(180)

        return img, total_calories, total_fat, total_carb, total_protein


class CombinedDataset(Dataset):

    def __init__(self, food101: Food101, nutrition5k: Nutrition5k):
        self.food101 = food101
        self.nutrition5k = nutrition5k

    def __len__(self):
        return len(self.food101) + len(self.nutrition5k)

    def __getitem__(self, index):
        if index < len(self.food101):
            has_nutrients = 0
            total_calories, total_fat, total_carb, total_protein = 0, 0, 0, 0
            img, target_class = self.food101.__getitem__(index)
        else:
            n5k_index = index - len(self.food101)
            has_nutrients = 1
            target_class = -1  # and make CrossEntropyLoss ignore this class by using ignore_index
            img, total_calories, total_fat, total_carb, total_protein = self.nutrition5k.__getitem__(
                n5k_index)

        nutrients = [total_calories, total_protein, total_carb, total_fat]

        return img, target_class, nutrients, has_nutrients


class FoodHead(nn.Module):

    def __init__(self, n_features=1280):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 101),  # 101 food classes
        )

        self.nutrition_predictor = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 4)  # calories, carbs, proteins and fato
        )

    def forward(self, features):
        predicted_class = self.classifier(features)
        predicted_nutrients = self.nutrition_predictor(features)

        return torch.cat([predicted_class, predicted_nutrients], dim=1)


def train_model():
    # train = Food101(root="./", split="train", download=True)
    # val = Food101(root="./", split="test", download=True)
    train, val = get_combined_datasets()
    os.makedirs(Settings.OUT_DIR, exist_ok=True)

    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    # model = mobilenet_v3_large(
    #     weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)

    model.classifier = FoodHead()
    model = model.to(Settings.DEVICE)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = AdamW([
        {"params": model.features.parameters(), "lr": 0.000005},
        {"params": model.classifier.parameters(), "lr": 0.0001}
    ])
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
    val_nutri_losses = []
    val_class_losses = []
    train_losses = []
    lowest_val_loss = float('inf')
    highest_accuracy = 0
    for e in range(n_epochs):
        print("*" * 10, "Epoch", e+1, "*" * 10)
        model.train()
        train_stats = train_one_epoch(
            model, train_dataloader, optimizer, loss_fn)

        print("Training:")
        for k, v in train_stats.items():
            if type(v) == float:
                print("\t", k, v)

        train_losses.append(train_stats["loss_sum"])

        model.eval()
        with torch.no_grad():
            val_stats = validate_one_epoch(model, val_dataloader, loss_fn)
            print("Validation:")

            for k, v in val_stats.items():
                if type(v) == float:
                    print("\t", k, v)

            confusion_matrix = val_stats["confusion_matrix"]
            num_correct = confusion_matrix.trace()
            total = confusion_matrix.sum()
            val_accuracy = num_correct / total
            print("val acc:", val_accuracy)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_stats["loss_sum"])
            val_nutri_losses.append(val_stats["nutri_loss_sum"])
            val_class_losses.append(val_stats["class_loss_sum"])

        plt.title("Loss")
        plt.plot(val_losses, label="validation")
        plt.plot(train_losses, label="training")
        plt.legend()
        plt.savefig(Settings.OUT_DIR / "loss.png")
        plt.close()

        plt.title("Validation loss components")
        plt.plot(val_nutri_losses, label="Nutrition prediction")
        plt.plot(val_class_losses, label="Class predictions")
        plt.legend()
        plt.savefig(Settings.OUT_DIR / "val_loss_components.png")
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
                f.write(
                    "train loss,val loss, val acc, val nutri loss, val class loss\n")

            f.write(
                f"{str(train_stats['loss_sum'])}, {str(val_stats['loss_sum'])}, {str(val_accuracy)}, {str(val_stats["nutri_loss_sum"])}, {str(val_stats["class_loss_sum"])}\n")


def get_combined_datasets():
    f101_train = Food101(root="./", split="train", download=True)
    f101_val = Food101(root="./", split="test", download=True)
    n5k_train, n5k_val = get_nutrition5k_datasets(val_ratio=0.2)

    train_dataset = CombinedDataset(f101_train, n5k_train)
    val_dataset = CombinedDataset(f101_val, n5k_val)

    return train_dataset, val_dataset


if __name__ == "__main__":
    # test_combined_dataset()

    train_model()
