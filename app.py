import os
import io
import base64
import uuid
import requests
import time
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
from datetime import datetime
import torch

# Set to use CPU only (for Render)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Folder setup
TEST_IMAGES_FOLDER = 'static/test_images'
RESULTS_FOLDER = 'static/results'
UPLOAD_FOLDER = 'static/uploads'

os.makedirs(TEST_IMAGES_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Download YOLOv5 model from Google Drive if not present
MODEL_PATH = "pcb_defect_model.pt"
GDRIVE_FILE_ID = "1-0Gwiux0iVPBQ34Ku9j2WmOtoz_TICdk"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

if not os.path.exists(MODEL_PATH):
    print("Downloading YOLOv5 model from Google Drive...")
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Model downloaded.")
        time.sleep(2)
    else:
        raise RuntimeError("Failed to download model.")

# Load YOLOv5 model
try:
    model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=False)
    print("YOLOv5 model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading YOLOv5 model: {e}")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def index():
    images = os.listdir(TEST_IMAGES_FOLDER)
    return render_template("index.html", images=images)

@app.route("/detect", methods=["POST"])
def detect():
    image_name = request.form["image"]
    image_path = os.path.join(TEST_IMAGES_FOLDER, image_name)

    results = model(image_path)
    results.render()

    result_path = os.path.join(RESULTS_FOLDER, image_name)
    Image.fromarray(results.ims[0]).save(result_path)

    return redirect(url_for("result", filename=image_name))

@app.route("/webcam")
def webcam():
    return render_template("webcam.html")

@app.route("/upload_webcam", methods=["POST"])
def upload_webcam():
    data = request.get_json()
    image_data = data["imageData"].split(",")[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"webcam_{timestamp}.jpg"
    input_path = os.path.join(TEST_IMAGES_FOLDER, filename)
    result_path = os.path.join(RESULTS_FOLDER, filename)

    image.save(input_path)

    results = model(input_path)
    results.render()
    Image.fromarray(results.ims[0]).save(result_path)

    return url_for("result", filename=filename)

@app.route("/result/<filename>")
def result(filename):
    result_url = url_for("static", filename=f"results/{filename}")
    return f"""
    <h2>Detection Result</h2>
    <img src="{result_url}" style="max-width:100%;"/>
    <br><a href="/">Back to Home</a>
    """

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # This must match your render.yaml PORT env var
    app.run(host="0.0.0.0", port=port)
