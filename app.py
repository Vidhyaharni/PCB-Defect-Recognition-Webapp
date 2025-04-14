import os
import sys
import uuid
import base64
import requests
import logging
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from PIL import ImageDraw
from flask import Flask, render_template, request

# Setup logs
logging.basicConfig(level=logging.INFO)

# Add YOLOv5 path
YOLOV5_PATH = Path(__file__).resolve().parent / 'yolov5'
sys.path.append(str(YOLOV5_PATH))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

app = Flask(__name__)

# Paths
ROOT_DIR = os.getcwd()
MODEL_PATH = os.path.join(ROOT_DIR, 'pcb_defect_model.pt')
TEST_IMAGES_FOLDER = os.path.join('static', 'test_images')
RESULTS_FOLDER = os.path.join('static', 'results')
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Google Drive model download
MODEL_DRIVE_ID = "16RButtzMCKMJmlCDR8XG42JII1gcj_HP"
GDRIVE_URL = f"https://drive.google.com/uc?export=download&id={MODEL_DRIVE_ID}"

if not os.path.exists(MODEL_PATH):
    logging.info("Downloading model...")
    try:
        r = requests.get(GDRIVE_URL)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        logging.info("✅ Model downloaded.")
    except Exception as e:
        logging.error(f"❌ Failed to download model: {e}")
        raise RuntimeError("Model download failed.")

# Load model
device = select_device("cpu")
model = DetectMultiBackend(MODEL_PATH, device=device)
model.eval()
logging.info("✅ Model loaded.")

# Detection function
def detect_image_yolov5(image_pil):
    try:
        img0 = np.array(image_pil.convert("RGB"))
        img = np.array(image_pil.resize((416, 416))).transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        im = torch.from_numpy(img).to(device)
        im = im.float() / 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.25, 0.45)

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in det:
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    x1, y1, x2, y2 = map(int, xyxy)
                    img0 = Image.fromarray(img0)
                    draw = ImageDraw.Draw(img0)
                    draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
                    draw.text((x1, y1 - 10), label, fill="green")
        return img0 if isinstance(img0, Image.Image) else Image.fromarray(img0)
    except Exception as e:
        logging.error(f"Detection failed: {e}")
        raise

@app.route('/')
def index():
    images = os.listdir(TEST_IMAGES_FOLDER)
    return render_template('index.html', images=images)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        image_name = request.form['image']
        image_path = os.path.join(TEST_IMAGES_FOLDER, image_name)
        image = Image.open(image_path)

        result_img = detect_image_yolov5(image)
        result_filename = f"{uuid.uuid4().hex}.jpg"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        result_img.save(result_path)

        return render_template('result.html', result_image=result_filename)
    except Exception as e:
        logging.error(f"/detect route failed: {e}")
        return "Error in /detect route", 500

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/upload_webcam', methods=['POST'])
def upload_webcam():
    try:
        data = request.get_json()
        image_data = data['imageData'].split(',')[1]
        image_bytes = base64.b64decode(image_data)

        image = Image.open(BytesIO(image_bytes))

        result_img = detect_image_yolov5(image)
        result_filename = f"{uuid.uuid4().hex}.jpg"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        result_img.save(result_path)

        return f"/result_from_webcam/{result_filename}"
    except Exception as e:
        logging.error(f"/upload_webcam route failed: {e}")
        return "Error in /upload_webcam route", 500

@app.route('/result_from_webcam/<result_image>')
def result_from_webcam(result_image):
    return render_template('result.html', result_image=result_image)

# Render compatible run block
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Default to 8080
    app.run(host='0.0.0.0', port=port)