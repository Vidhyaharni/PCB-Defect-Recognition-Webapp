from flask import Flask, render_template, request, jsonify
import os
<<<<<<< HEAD
import uuid
import pathlib
import base64
from io import BytesIO

# Fix PosixPath issue on Windows
class WindowsPath(pathlib.PosixPath):
    def __new__(cls, *args, **kwargs):
        return pathlib.WindowsPath(*args, **kwargs)

pathlib.PosixPath = WindowsPath

try:
    import torch
    from PIL import Image
except ModuleNotFoundError as e:
    raise RuntimeError("Required modules not found. Please ensure 'torch' and 'Pillow' are installed.") from e

app = Flask(__name__)

# Load YOLOv5 model
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/vidhya/Documents/pcb-defect-app/pcb_defect_model.pt', force_reload=True)
except Exception as e:
    raise RuntimeError("Error loading YOLOv5 model. Ensure model.pt is in the correct path and torch is properly installed.") from e

# Folders
TEST_IMAGES_FOLDER = 'static/test_images'
RESULTS_FOLDER = 'static/results'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    images = os.listdir(TEST_IMAGES_FOLDER)
    return render_template('index.html', images=images)

@app.route('/detect', methods=['POST'])
def detect():
    image_name = request.form['image']
    image_path = os.path.join(TEST_IMAGES_FOLDER, image_name)

    results = model(image_path)
    result_img = results.render()[0]
    result_pil = Image.fromarray(result_img)

    result_filename = f"{uuid.uuid4().hex}.jpg"
    result_path = os.path.join(RESULTS_FOLDER, result_filename)
    result_pil.save(result_path)

    return render_template('result.html', result_image=result_filename)
=======
import io
import gdown
from PIL import Image
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Ensure YOLOv5 model is downloaded from Google Drive
def download_model():
    file_id = "1-0Gwiux0iVPBQ34Ku9j2WmOtoz_TICdk"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "pcb_defect_model.pt"
    if not os.path.exists(output):
        print("Downloading model from Google Drive...")
        gdown.download(url, output, quiet=False)

download_model()

# Load the model (CPU only for Render)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='pcb_defect_model.pt', force_reload=True)
model.to('cpu')

# Home page with test images
@app.route('/')
def index():
    test_images = os.listdir('static/test_images')
    return render_template('index.html', test_images=test_images)
>>>>>>> 5d53de6 (Update app.py and requirements.txt for Render deployment)

# Webcam upload page
@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

<<<<<<< HEAD
    image_data = data['imageData']
    header, encoded = image_data.split(",", 1)
    binary_data = base64.b64decode(encoded)
=======
    file = request.files['image']
    filename = secure_filename(file.filename)
    img_bytes = file.read()
>>>>>>> 5d53de6 (Update app.py and requirements.txt for Render deployment)

    # Convert image bytes to PIL image
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

<<<<<<< HEAD
    results = model(filepath)
    result_img = results.render()[0]
    result_pil = Image.fromarray(result_img)
=======
    # Run YOLOv5 model
    results = model(image)
>>>>>>> 5d53de6 (Update app.py and requirements.txt for Render deployment)

    # Save image with detections
    output_path = os.path.join('static', 'results', filename)
    results.save(save_dir='static/results')  # Saves into static/results directory

    return jsonify({'result_path': f'/static/results/{filename}'})

if __name__ == '__main__':
    app.run(debug=True)
