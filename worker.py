import requests
import time
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor


class Settings:
    API_BASE_URL = "http://localhost:5000/api"
    WORK_URL = f"{API_BASE_URL}/food-work"
    FOOD_URL = f"{API_BASE_URL}/food"
    DEVICE = "cpu"


def get_food_image(id) -> Image.Image:
    return Image.open(requests.get(f"{Settings.FOOD_URL}/{id}/image", stream=True).raw)


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


class MobileNetModelWorker:

    def __init__(self):
        self.model = torch.jit.load(
            "model.pt", map_location=torch.device("cpu")).to(Settings.DEVICE)
        self.class_to_label_map = load_label_map()

    def work_one_cycle(self):
        food_ids = get_work()

        if len(food_ids) == 0:
            return

        for id in food_ids:
            self.process_food(id)

    def process_food(self, id):
        food_img = get_food_image(id)

        food_img = food_img.resize((224, 224)).convert("RGB")

        # convert to tensor and normalize it
        img_tensor = pil_to_tensor(food_img).type(torch.float32)
        img_tensor -= 256/2
        img_tensor /= 256

        batch = torch.stack([img_tensor]).to(Settings.DEVICE)

        pred = self.model(batch)

        predicted_class = pred.argmax(1)

        food_type = self.class_to_label_map[predicted_class.item()]

        analysis_result = {
            "food_type": food_type
        }

        send_food_analysis_result(id, analysis_result)


def work(worker: MobileNetModelWorker):
    while True:
        try:
            worker.work_one_cycle()
        except Exception as e:
            print("Fail:", e)
            time.sleep(5)


if __name__ == "__main__":
    worker = MobileNetModelWorker()
    work(worker)
