import sqlite3
from uuid import uuid4
from PIL import Image
import io

sqlite3.SQLITE_CONSTRAINT_FOREIGNKEY = 1


def get_connection():
    return sqlite3.connect("food.db")


def init_db():
    conn = sqlite3.connect("food.db")
    create_food_table_sql = """
    CREATE TABLE food(
        id TEXT PRIMARY KEY,
        image BLOB,
        status TEXT
    );
    """

    create_food_analysis_table_sql = """
    CREATE TABLE food_analysis(
        food_type TEXT, 
        food_id INTEGER, 
        FOREIGN KEY(food_id) REFERENCES food(id)
    );
    """
    conn.execute(create_food_table_sql)
    conn.execute(create_food_analysis_table_sql)


def insert_food(img: Image.Image):
    conn = get_connection()
    id = str(uuid4())
    bytes = io.BytesIO()
    img.save(bytes, format="PNG")
    conn.execute(
        f"INSERT INTO food(id, image, status) VALUES (?, ?, ?)", (id, bytes.getvalue(), "CREATED"))

    conn.commit()
    conn.close()

    return id


def get_food_status(id):
    conn = get_connection()
    cursor = conn.execute("SELECT status FROM food WHERE id = (?)", (id,))

    status = cursor.fetchone()[0]

    return status


def get_food_img(id):
    conn = get_connection()
    cursor = conn.execute("SELECT image FROM food WHERE id = (?)", (id,))
    image_blob = cursor.fetchone()[0]
    image_stream = io.BytesIO(image_blob)
    image = Image.open(image_stream)
    return image


def insert_food_img_analysis(id, food_type):
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO food_analysis(food_id, food_type) VALUES (?, ?)", (id, food_type))

    conn.commit()



def get_food_img_analysis(id):
    conn = get_connection()
    cursor = conn.execute(
        "SELECT food_type FROM food_analysis WHERE food_id = (?)", (id,))
    food_type = cursor.fetchone()

    return food_type


def test():
    img = Image.open("egg_sandwhich.png")
    id = insert_food(img)
    food = get_food_status(id)
    fetched_img = get_food_img(id)

    insert_food_img_analysis(id, "SALAD")
    food_type = get_food_img_analysis(id)
    print(food_type)


if __name__ == "__main__":
    init_db()
    # test()
