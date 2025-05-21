import time

import cv2
from flask import Flask, Response

app = Flask(__name__)


def generate_frames():
    while True:
        try:
            # Öffne das aktuelle Bild (z. B. von einer Kamera oder einem Skript erzeugt)
            frame = cv2.imread("latest.jpg")  # <- ersetze durch Euer Bild
            if frame is None:
                time.sleep(0.1)
                continue

            # Kodieren als JPEG
            ret, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()

            # MJPEG-Frame senden
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.05)  # ~20 FPS

        except Exception as e:
            print("Fehler beim Laden des Bildes:", e)
            time.sleep(0.5)


@app.route("/")
def index():
    return "🟢 MJPEG-Stream läuft: Öffne /video_feed im Browser"


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
