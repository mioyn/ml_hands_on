# python -m pip install opencv-python
import math

import cv2
import numpy as np

MAXX, MAXY = 800, 800


def zigzag(start, end):
    i = start
    while True:
        while i < end:
            i += 1
            yield i
        while i > start:
            i -= 1
            yield i


def sinewave(start, end, frequency, phase=0.0):
    i = phase
    while True:
        i += frequency
        yield int((end - start) * (math.sin(i) + 1) / 2 + start)


xgen = sinewave(start=0, end=600, frequency=math.pi / 360)
ygen = sinewave(start=0, end=600, frequency=math.pi / 313)
blue = zigzag(start=0, end=255)
green = zigzag(start=49, end=255)
red = zigzag(start=64, end=255)

while True:
    frame = np.zeros((MAXY, MAXX, 3), np.uint8)
    x = next(xgen)
    y = next(ygen)
    frame[y : y + 50, x : x + 50] = (next(blue), next(green), next(red))

    cv2.imshow("frame", frame)
    key = chr(cv2.waitKey(1) & 0xFF)

    if key == "q":
        break

cv2.destroyAllWindows()
