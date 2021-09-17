import cv2
import numpy as np
from matplotlib import pyplot as plt
import utilities

def main():
    vc = cv2.VideoCapture('background.mp4')
    if vc.isOpened():
        print("Video opened!")
    else:
        print("Video open error!")

    # Read the first frame as background
    ret, frame = vc.read()
    gray_frame = 1
    