from flask import Flask, Response
from flask_cors import CORS
from ultralytics import YOLO
import cv2

app = Flask(__name__)
CORS(app)

model = YOLO("A+B+O+L+P+GFreshnessDetection.pt")

# Simulate webcam feed — this will later receive frames from another system (System A)
cap = cv2.VideoCapture(0)  # Replace this with a video stream if needed

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLOv8 detection
        results = model(frame)
        annotated_frame = results[0].plot()

        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # Yield as MJPEG frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def home():
    return "🔥 Freshness Detection MJPEG Stream is Running!"

@app.route('/stream')
def stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
