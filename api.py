import database
from flask import Flask, request, jsonify, abort, send_file, make_response
from PIL import Image
from io import BytesIO
import time

app = Flask(__name__)

base_path = "/api"


@app.route(f'{base_path}/status')
def get_api_status():
    return "api is doing great!"


@app.route(f'{base_path}/food-work')
def get_work():
    long_poll_wait_seconds = 30
    time_start = time.time()
    while True:
        ids = database.get_unprocessed_foods()
        if len(ids) > 0:
            break

        if time.time() - time_start >= long_poll_wait_seconds:
            break

        time.sleep(0.1)

    # Returns list of food images (ids) that are not processed
    return jsonify({
        "food_ids": ids
    })


@app.route(f"{base_path}/food/<id>")
def get_food(id: str):
    id, status = database.get_food_status(id)

    return jsonify({
        "id": id,
        "status": status
    })


@app.route(f"{base_path}/food/<id>/image")
def get_food_image(id: str):
    img = database.get_food_img(id)
    
    img_io = BytesIO()
    
    img.save(img_io, 'PNG')

    img_io.seek(0)


    return send_file(img_io, mimetype='image/png')


@app.route(f"{base_path}/food/<id>/analysis_result", methods=["POST"])
def post_food_analysis(id: str):
    food_type = request.json.get("food_type")
    database.insert_food_img_analysis(id, food_type)
    print("Image processed", time.time())
    return ""


@app.route(f"{base_path}/food/<id>/analysis_result")
def get_food_analysis(id: str):
    time_start = time.time()

    # long poll
    max_wait_seconds = 30
    while True:
        print(id)
        status = database.get_food_status(id)
        if status == "COMPLETED":
            break
        time.sleep(0.1)

        if time.time() - time_start >= max_wait_seconds:
            abort(504)

    food_type = database.get_food_img_analysis(id)

    return jsonify({
        "food_type": food_type
    })


@app.route(f'{base_path}/food', methods=['POST'])
def request_food_content():
    print("Image received", time.time())
    file = request.files['image']
    img = Image.open(file).copy() 

    id = database.insert_food(img)

    return jsonify({
        "id": id
    })


if __name__ == '__main__':
    app.run(
        debug=True,
        threaded=True
    )
