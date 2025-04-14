import os
import sys
import uuid
import base64
import requests
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request
from PIL import Image
from pathlib import Path

# Add YOLOv5 path for Render & local
YOLOV5_DIR = Path(__file__).resolve().parent / "yolov5"
sys.path.append(str(YOLOV5_DIR))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

app = Flask(__name__)

# Paths
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "pcb_defect_model.pt"
STATIC = ROOT / "static"
TEST_IMAGES_FOLDER = STATIC / "test_images"
RESULTS_FOLDER = STATIC / "results"
UPLOAD_FOLDER = STATIC / "uploads"
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Auto-download model if missing
GDRIVE_FILE_ID = "1-0Gwiux0iVPBQ34Ku9j2WmOtoz_TICdk"
if not MODEL_PATH.exists():
    print("📦 Downloading YOLOv5 model...")
    gdrive_url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
    response = requests.get(gdrive_url)
    if response.ok:
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded.")
    else:
        raise RuntimeError("❌ Model download failed!")

# ✅ Load model once
device = select_device("cpu")
model = DetectMultiBackend(weights=str(MODEL_PATH), device=device)
model.eval()
print("✅ Model loaded.")

# 🔍 Detection function
def detect_image(image_path):
    img0 = cv2.imread(str(image_path))  # Original
    img = cv2.resize(img0, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    im = torch.from_numpy(img).to(device).float() / 255.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)

    pred = model(im, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.25, 0.45)

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                label = f"{model.names[int(cls)]} {conf:.2f}"
                print(f"🟢 Prediction: {label}")
                cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), 
                                   (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print("⚠️ No defects detected in image.")

    return Image.fromarray(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))

@app.route("/")
def index():
    images = os.listdir(TEST_IMAGES_FOLDER)
    return render_template("index.html", images=images)

@app.route("/detect", methods=["POST"])
def detect():
    try:
        image_name = request.form["image"]
        image_path = TEST_IMAGES_FOLDER / image_name

        result_img = detect_image(image_path)
        result_filename = f"{uuid.uuid4().hex}.jpg"
        result_path = RESULTS_FOLDER / result_filename
        result_img.save(result_path)

        return render_template("result.html", result_image=result_filename)
    except Exception as e:
        print("❌ Error during detection:", e)
        return "Error occurred during detection", 500

@app.route("/webcam")
def webcam():
    return render_template("webcam.html")

@app.route("/upload_webcam", methods=["POST"])
def upload_webcam():
    try:
        data = request.get_json()
        image_data = data["imageData"].split(",")[1]
        image_bytes = base64.b64decode(image_data)

        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = UPLOAD_FOLDER / filename
        with open(filepath, "wb") as f:
            f.write(image_bytes)

        result_img = detect_image(filepath)
        result_filename = f"{uuid.uuid4().hex}.jpg"
        result_path = RESULTS_FOLDER / result_filename
        result_img.save(result_path)

        return f"/result_from_webcam/{result_filename}"
    except Exception as e:
        print("❌ Webcam upload error:", e)
        return "Error during webcam upload", 500

@app.route("/result_from_webcam/<result_image>")
def result_from_webcam(result_image):
    return render_template("result.html", result_image=result_image)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
