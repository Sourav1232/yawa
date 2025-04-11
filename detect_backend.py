from flask import Flask, request, send_file
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO

app = Flask(__name__)
CORS(app, resources={r"/detect": {"origins": "https://foodapp-seven-self.vercel.app"}})

print("✅ Loading YOLO model...")
model = YOLO("A+B+O+L+P+GFreshnessDetection.pt")
print("✅ Model loaded successfully")

@app.route('/')
def home():
    print("📡 Received GET request at '/'")
    return "🔥 Freshness Detection API is Running!"

@app.route('/detect', methods=['POST'])
def detect():
    print("📩 Received POST request at '/detect'")
    if 'frame' not in request.files:
        print("❌ No frame found in request")
        return "No frame uploaded", 400

    try:
        file = request.files['frame']
        img_bytes = file.read()
        print(f"📦 Received frame of size {len(img_bytes)} bytes")
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        results = model(img)
        print("✅ YOLO model inference done")
        annotated_img = results[0].plot()

        _, buffer = cv2.imencode('.jpg', annotated_img)
        print("✅ Processed image encoded successfully")

        return send_file(BytesIO(buffer), mimetype='image/jpeg')
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        return "Error processing frame", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
