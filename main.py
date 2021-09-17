import numpy as np
import pandas as pd
import cv2
from utilities import process_frame, get_bin_image

def bubble_detector(stats, centroid, th=40):
    """
    A simple bubble detector. If a connected component enters the 
    detection area and its area is greater than TH, it would be 
    detected.

    :stats:         stats returned from cv2.connectedCom..
    :centroid:      position of bubble center
    :th:            bubble area threshold
    """

def main():

    # Process background: 720*1080 uint8
    # bg_frame = cv2.imread('mean-background.png')
    # bg_edge = process_frame(bg_frame)

    # Read video
    vc = cv2.VideoCapture('output.mp4')
    if vc.isOpened():
        print("Video opened!")
    else:
        print("Video open error!")

    ret, bg_frame = vc.read()
    bg_edge = process_frame(bg_frame)

    # Parameter Settings
    FPS = vc.get(cv2.CAP_PROP_FPS)
    HEIGHT = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    WIDTH = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    LENGTH = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    DETECTION_LOW_LIM = 500     # the line below
    DETECTION_UP_LIM = 430      # the line above

    bubble_counter = 0          # global count of bubblesk
    detected_bubble = pd.DataFrame(columns=['stats', 'centroid', 'is_counted'])
    

    while ret:
        ret, frame = vc.read()

        # Get bubble binary image
        edge_frame = process_frame(frame)
        bin_bubble_frame = get_bin_image(edge_frame, bg_edge)

        # Perform connected component analysis
        output = cv2.connectedComponentsWithStats(
            bin_bubble_frame, connectivity=8, ltype=cv2.CV_32S)
        num_labels, labels, stats, centroids = output

        for label in range(num_labels):
            if DETECTION_LOW_LIM > centroids[label, 1] and centroids[label, 1] > DETECTION_UP_LIM and stats[label, 4] > 40:


                cv2.rectangle(frame, (stats[label, 0:2]), (stats[label, 0:2]+stats[label, 2:4]), (0, 0, 255), 2)
        
        # Draw detection line below
        cv2.line(frame, (0, DETECTION_LOW_LIM), (WIDTH, DETECTION_LOW_LIM), (0, 0, 255), 1)

        # Draw detection line above
        cv2.line(frame, (0, DETECTION_UP_LIM), (WIDTH, DETECTION_UP_LIM), (0, 0, 255), 1)

        # Display text information
        cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putText(frame, str(vc.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv2.imshow('Frame', frame)
        cv2.imshow('Bubble Edge', bin_bubble_frame)

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break


if __name__ == '__main__':
    main()
