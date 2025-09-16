import requests
import time
from PIL import Image
import torch

API_BASE_URL = "http://localhost:5000/api"
WORK_URL = f"{API_BASE_URL}/food-work"
FOOD_URL = f"{API_BASE_URL}/food"


def get_food_image(id):
    return Image.open(requests.get(f"{FOOD_URL}/{id}/image", stream=True).raw)


def send_food_analysis_result(id, analysis_result):
    requests.post(f"{FOOD_URL}/{id}/analysis_result", json=analysis_result)


def process_food(id):
    print("Processing", id)
    food_img = get_food_image(id)

    # resize img
    # convert t to tensor
    # pass through model
    # get result

    analysis_result = {
        "food_type": "egg"
    }

    send_food_analysis_result(id, analysis_result)


def work_cycle():
    result = requests.get(WORK_URL)

    food_ids = result.json()["food_ids"]
    print("foods to be processed:", food_ids)
    if len(food_ids) == 0:
        return

    for id in food_ids:
        process_food(id)


def work():
    while True:
        try:
            work_cycle()
        except Exception as e:
            print("Fail:", e)
            time.sleep(5)


if __name__ == "__main__":
    work()
