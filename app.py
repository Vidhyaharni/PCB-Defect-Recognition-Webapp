import pathlib
import os
import uuid
import base64
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request
from PIL import Image
import sys
from pathlib import Path

# Patch to fix PosixPath issue on Windows
if pathlib.PosixPath != pathlib.WindowsPath:
    class WindowsPath(pathlib.PosixPath):
        def __new__(cls, *args, **kwargs):
            return pathlib.WindowsPath(*args, **kwargs)
    pathlib.PosixPath = WindowsPath

# Add yolov5 to path
sys.path.append(str(Path(__file__).resolve().parent / "yolov5"))
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

app = Flask(__name__)

# Folders
TEST_IMAGES_FOLDER = 'static/test_images'
RESULTS_FOLDER = 'static/results'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
device = select_device('cpu')
model = DetectMultiBackend('pcb_defect_model.pt', device=device)
model.eval()

def detect_image(image_path):
    img0 = cv2.imread(image_path)  # BGR
    img = cv2.resize(img0, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # to 3x640x640 RGB
    img = np.ascontiguousarray(img)

    im = torch.from_numpy(img).to(device).float() / 255.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)

    pred = model(im, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.25, 0.45, None, False)

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

@app.route('/')
def index():
    images = os.listdir(TEST_IMAGES_FOLDER)
    return render_template('index.html', images=images)

@app.route('/detect', methods=['POST'])
def detect():
    image_name = request.form['image']
    image_path = os.path.join(TEST_IMAGES_FOLDER, image_name)
    result_image = detect_image(image_path)

    result_filename = f"{uuid.uuid4().hex}.jpg"
    result_path = os.path.join(RESULTS_FOLDER, result_filename)
    result_image.save(result_path)

    return render_template('result.html', result_image=result_filename)

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/upload_webcam', methods=['POST'])
def upload_webcam():
    data = request.get_json()
    image_data = data['imageData'].split(',')[1]
    binary_data = base64.b64decode(image_data)

    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, 'wb') as f:
        f.write(binary_data)

    result = detect_image(filepath)
    result_filename = f"{uuid.uuid4().hex}.jpg"
    result_path = os.path.join(RESULTS_FOLDER, result_filename)
    result.save(result_path)

    return f"/result_from_webcam/{result_filename}"

@app.route('/result_from_webcam/<result_image>')
def result_from_webcam(result_image):
    return render_template('result.html', result_image=result_image)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)