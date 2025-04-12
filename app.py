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

# ✅ Add yolov5 to sys.path
YOLOV5_PATH = Path(__file__).parent / "yolov5"
if YOLOV5_PATH.exists():
    sys.path.insert(0, str(YOLOV5_PATH))
else:
    raise RuntimeError(f"❌ yolov5 folder not found at {YOLOV5_PATH}")

# ✅ YOLOv5 imports
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# ✅ Flask app setup
app = Flask(__name__)

# ✅ Paths
ROOT_DIR = os.getcwd()
MODEL_PATH = os.path.join(ROOT_DIR, 'pcb_defect_model.pt')
TEST_IMAGES_FOLDER = os.path.join('static', 'test_images')
RESULTS_FOLDER = os.path.join('static', 'results')
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(TEST_IMAGES_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Download model from Google Drive if missing
GDRIVE_FILE_ID = "1-0Gwiux0iVPBQ34Ku9j2WmOtoz_TICdk"
GDRIVE_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
if not os.path.exists(MODEL_PATH):
    print("🔄 Downloading model from Google Drive...")
    r = requests.get(GDRIVE_URL)
    if r.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("✅ Model downloaded.")
    else:
        raise RuntimeError("❌ Failed to download model.")

# ✅ Load YOLOv5 model
device = select_device("cpu")
model = DetectMultiBackend(MODEL_PATH, device=device)
model.eval()

# ✅ Detection Function
def detect_image_yolov5(image_path):
    img0 = cv2.imread(image_path)
    img = cv2.resize(img0, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)
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
                cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

# ✅ Routes
@app.route('/')
def index():
    images = os.listdir(TEST_IMAGES_FOLDER)
    return render_template('index.html', images=images)

@app.route('/detect', methods=['POST'])
def detect():
    image_name = request.form['image']
    image_path = os.path.join(TEST_IMAGES_FOLDER, image_name)
    result_img = detect_image_yolov5(image_path)

    result_filename = f"{uuid.uuid4().hex}.jpg"
    result_path = os.path.join(RESULTS_FOLDER, result_filename)
    result_img.save(result_path)

    return render_template('result.html', result_image=result_filename)

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/upload_webcam', methods=['POST'])
def upload_webcam():
    data = request.get_json()
    image_data = data['imageData'].split(',')[1]
    image_bytes = base64.b64decode(image_data)

    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    result_img = detect_image_yolov5(filepath)
    result_filename = f"{uuid.uuid4().hex}.jpg"
    result_path = os.path.join(RESULTS_FOLDER, result_filename)
    result_img.save(result_path)

    return f"/result_from_webcam/{result_filename}"

@app.route('/result_from_webcam/<result_image>')
def result_from_webcam(result_image):
    return render_template('result.html', result_image=result_image)

# ✅ Start app (Render needs this format)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
