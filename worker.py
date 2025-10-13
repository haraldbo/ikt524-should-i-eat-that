import requests
import time
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor
import io
import torchvision.transforms as transforms


class Settings:
    API_BASE_URL = "http://127.0.0.1:5000/api"
    WORK_URL = f"{API_BASE_URL}/food-work"
    FOOD_URL = f"{API_BASE_URL}/food"
    DEVICE = "cpu"


class Model:

    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_prediction(self, img: Image.Image):
        raise NotImplementedError()


class Food101PredictionResult:

    def __init__(self):
        self.class_to_label_map = load_label_map()

    def __call__(self, pred: torch.Tensor):
        probs = pred.softmax(1)

        predicted_class = pred.argmax(1)

        food_type = self.class_to_label_map[predicted_class.item()]

        probalibity = round(probs[0, predicted_class].item(), 3)

        return {
            "food_type": food_type,
            "confidence": probalibity
        }


class MobileNet(Model):

    def __init__(self):
        super().__init__("mobile net")
        self.model = torch.jit.load(
            "./trained_models/mobile_net/model.pt", map_location=torch.device(Settings.DEVICE)).to(Settings.DEVICE)
        self.food101_prediction_results = Food101PredictionResult()

    def get_prediction(self, img):
        img = img.resize((224, 224)).convert("RGB")

        # convert to tensor and normalize it
        img_tensor = pil_to_tensor(img).type(torch.float32)
        img_tensor -= 256/2
        img_tensor /= 256

        batch = torch.stack([img_tensor]).to(Settings.DEVICE)

        pred = self.model(batch)

        return self.food101_prediction_results(pred)


class EfficientNet(Model):

    def __init__(self):
        super().__init__("EFficientNet")
        self.model = torch.jit.load(
            "./trained_models/efficient_net/model.pt", map_location=torch.device(Settings.DEVICE)).to(Settings.DEVICE)
        self.food101_prediction = Food101PredictionResult()

    def get_prediction(self, img):
        img = img.convert("RGB")
        img_tensor = transforms.Compose([
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]  # a bit weird values?
            )
        ])(img)

        batch = torch.stack([img_tensor]).to(Settings.DEVICE)

        pred = self.model(batch)

        return self.food101_prediction(pred)


def get_food_image(id) -> Image.Image:
    print("make request", time.time())
    response = requests.get(f"{Settings.FOOD_URL}/{id}/image")
    response.raw.chunked = True
    print("request received", time.time())
    img_data = io.BytesIO(response.content)
    img = Image.open(img_data)
    print("Got image", time.time())
    return img


def send_food_analysis_result(id, analysis_result):
    requests.post(f"{Settings.FOOD_URL}/{id}/analysis_result",
                  json=analysis_result)


def get_work():
    result = requests.get(Settings.WORK_URL)
    food_ids = result.json()["food_ids"]
    return food_ids


def load_label_map():
    file = open("./food-101/meta/labels.txt", "r")

    labels = {}
    for i, label in enumerate(file.readlines()):
        labels[i] = label.strip()

    file.close()
    return labels


class Worker:

    def __init__(self, model: Model):
        print("Using model", model.model_name)
        self.model = model

    def work_one_cycle(self):
        print("Wait for work")
        food_ids = get_work()

        if len(food_ids) == 0:
            return

        for id in food_ids:
            self.process_food(id)

    def process_food(self, id):
        food_img = get_food_image(id)

        prediction = self.model.get_prediction(food_img)

        send_food_analysis_result(id, prediction)


def work(worker: Worker):
    while True:
        try:
            worker.work_one_cycle()
        except Exception as e:
            print("Fail:", e)
            time.sleep(5)


if __name__ == "__main__":
    worker = Worker(EfficientNet())
    work(worker)
