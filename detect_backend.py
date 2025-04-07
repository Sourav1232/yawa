from flask import Flask, request, send_file
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO

app = Flask(__name__)
CORS(app)

model = YOLO("A+B+O+L+P+GFreshnessDetection.pt")

@app.route('/')
def home():
    return "🔥 Freshness Detection API is Running!"

@app.route('/detect', methods=['POST'])
def detect():
    if 'frame' not in request.files:
        return "No frame uploaded", 400

    file = request.files['frame']
    img_bytes = file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = model(img)
    annotated_img = results[0].plot()

    _, buffer = cv2.imencode('.jpg', annotated_img)
    return send_file(BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
