import time

import cv2
from flask import Flask, Response, redirect, url_for

from shared_src.common.threading import StoppableThread

app = Flask(__name__)


def generate_frames():
    while True:
        try:
            frame = cv2.imread("latest.jpg")
            if frame is None:
                time.sleep(0.1)
                continue
            ret, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.05)
        except Exception as e:
            print("Fehler beim Laden des Bildes:", e)
            time.sleep(0.5)


@app.route("/")
def index():
    return redirect(url_for("live_display"))


@app.route("/live_display")
def live_display():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


class DisplayServer(StoppableThread):
    def __init__(self, port: int, **kwargs):
        """
        Initialize the DisplayServer.
        Args:
            **kwargs: Additional arguments to pass to the StoppableThread.
        """

        super().__init__(**kwargs)
        self._module = __name__.split(".")[-1]
        self._process = f"gunicorn -b 0.0.0.0:5000 {self._module}:app"
