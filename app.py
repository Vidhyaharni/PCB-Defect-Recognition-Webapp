import os
import io
import base64
import uuid
import requests
import time
import sys
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image, ImageDraw
from datetime import datetime
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Use CPU only (for Render)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Add local yolov5 path
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

# YOLOv5 imports
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.datasets import letterbox

# Folder setup
TEST_IMAGES_FOLDER = 'static/test_images'
RESULTS_FOLDER = 'static/results'
UPLOAD_FOLDER = 'static/uploads'

os.makedirs(TEST_IMAGES_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model settings
MODEL_PATH = "pcb_defect_model.pt"
GDRIVE_FILE_ID = "1-0Gwiux0iVPBQ34Ku9j2WmOtoz_TICdk"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

# Download model if missing
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

# Load model
device = select_device('cpu')
model = attempt_load(MODEL_PATH, map_location=device)
print("YOLOv5 model loaded successfully.")

# Flask app
app = Flask(__name__)

def detect_image(image_path, save_path):
    img = Image.open(image_path).convert("RGB")
    img_resized = letterbox(np.array(img), new_shape=640)[0]
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    pred = model(img_tensor.to(device), augment=False)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img.size).round()

    model.names = model.names if hasattr(model, 'names') else model.module.names
    img_with_boxes = img.copy()
    draw = ImageDraw.Draw(img_with_boxes)

    for *xyxy, conf, cls in det:
        draw.rectangle(xyxy, outline="red", width=2)
        draw.text((xyxy[0], xyxy[1]), f"{model.names[int(cls)]} {conf:.2f}", fill="red")

    img_with_boxes.save(save_path)

@app.route("/")
def index():
    images = os.listdir(TEST_IMAGES_FOLDER)
    return render_template("index.html", images=images)

@app.route("/detect", methods=["POST"])
def detect():
    image_name = request.form["image"]
    image_path = os.path.join(TEST_IMAGES_FOLDER, image_name)
    result_path = os.path.join(RESULTS_FOLDER, image_name)

    detect_image(image_path, result_path)
    return redirect(url_for("result", filename=image_name))

@app.route("/webcam")
def webcam():
    return render_template("webcam.html")

@app.route("/upload_webcam", methods=["POST"])
def upload_webcam():
    data = request.get_json()
    image_data = data["imageData"].split(",")[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"webcam_{timestamp}.jpg"
    input_path = os.path.join(TEST_IMAGES_FOLDER, filename)
    result_path = os.path.join(RESULTS_FOLDER, filename)

    image.save(input_path)
    detect_image(input_path, result_path)

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
