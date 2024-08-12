import cv2
import numpy
from mss import mss

ScreenX = 1920
ScreenY = 1080
window_size = (
    int(ScreenX / 2 - 220),
    int(ScreenY / 2 - 220),
    int(ScreenX / 2 + 220),
    int(ScreenY / 2 + 220))
ScreenShot_value = mss()

def screenshot():
    img = ScreenShot_value.grab(window_size)
    img = numpy.array(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
    return img

# while True:
#     cv2.imshow('a',numpy.array(screenshot()))
#     cv2.waitKey(1)