import argparse
import time
from datetime import datetime

import cv2
import numpy as np

# !/usr/bin/env python3
"""
opencvdemo.py - simple OpenCV sample app

Usage:
    python opencvdemo.py           # use webcam
    python opencvdemo.py --image path/to/image.jpg
Keys:
    s - save current frame
    q or ESC - quit
"""


def int_or_default(x, default):
    try:
        return int(x)
    except (ValueError, TypeError):
        return default


def make_odd(x):
    x = int_or_default(x, 1)
    return x if x % 2 == 1 and x > 0 else max(1, x + 1)


def save_image(img):
    fname = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(fname, img)
    print("Saved:", fname)


def nothing(x):
    pass


def run(image_path=None):
    use_cam = image_path is None
    cap = None
    frame = None

    if use_cam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera. Falling back to sample image.")
            use_cam = False

    if not use_cam:
        if image_path:
            frame = cv2.imread(image_path)
        if frame is None:
            # generate a synthetic sample if no image provided
            frame = np.full((480, 640, 3), 200, dtype=np.uint8)
            cv2.putText(
                frame,
                "No camera/image - synthetic sample",
                (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (10, 10, 10),
                2,
            )

    # Create windows and trackbars
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("CannyLow", "Edges", 50, 500, nothing)
    cv2.createTrackbar("CannyHigh", "Edges", 150, 500, nothing)
    cv2.createTrackbar("Blur", "Edges", 3, 31, nothing)

    prev_time = time.time()
    while True:
        if use_cam:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

        # Processing
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Trackbar values
        low = cv2.getTrackbarPos("CannyLow", "Edges")
        high = cv2.getTrackbarPos("CannyHigh", "Edges")
        k = make_odd(cv2.getTrackbarPos("Blur", "Edges"))

        blurred = cv2.GaussianBlur(gray, (k, k), 0)
        edges = cv2.Canny(blurred, max(1, low), max(1, high))

        # Draw overlay on original
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (220, 60), (0, 0, 0), -1)
        fps = 1.0 / max(1e-6, (time.time() - prev_time))
        prev_time = time.time()
        cv2.putText(
            overlay,
            f"{w}x{h} FPS:{fps:.1f}",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        # Show a center crosshair
        cv2.drawMarker(
            overlay,
            (w // 2, h // 2),
            (0, 0, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=20,
            thickness=1,
        )

        # Show windows
        cv2.imshow("Original", overlay)
        cv2.imshow("Edges", edges)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            # Save the original overlay and edges side-by-side
            combined = np.hstack(
                [cv2.resize(overlay, (w, h)), cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)]
            )
            save_image(combined)
        if key == ord("q") or key == 27:  # ESC
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Simple OpenCV demo")
    p.add_argument("--image", "-i", help="path to image file (optional)", default=None)
    args = p.parse_args()
    run(args.image)
