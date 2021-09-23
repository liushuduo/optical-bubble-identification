import numpy as np
import pandas as pd
import cv2
from scipy.spatial.distance import cdist

from utilities import process_frame, get_bin_image, bubble_detector

# global constants
BUBBLE_MAXIMUM_MOVEMENT = 30
BUBBLE_COUNTER = 0

def update_bubble(new_bubble_table, old_bubble_table):
    
    if  old_bubble_table.empty:
        # If there is no bubbles detected before, all new detected bubbles are uncounted
        new_bubble_table['is_counted'] = False
        return

    for new_id, new_bubble in new_bubble_table.iterrows():

        old_bubble_table_temp = old_bubble_table.copy()

        # Compute the azumith between any old bubble and new bubble

        temp_a = np.stack(old_bubble_table['centroid'].to_numpy()) 
        temp_b = new_bubble['centroid']
        temp_a
        temp_b
        relative_coordinate = temp_a - temp_b

        # relative_coordinate = old_bubble_table['centroid'].to_numpy() - new_bubble['centroid']
        old_bubble_table_temp['azumith'] = np.arctan2(relative_coordinate[:, 1], relative_coordinate[:, 0])
        old_bubble_table_temp['distance'] = np.linalg.norm(relative_coordinate, axis=1)
        
        # Find satisfied bubbles
        satisfied_bubbles = old_bubble_table_temp[
                            (old_bubble_table_temp['azumith'] < np.pi/3*2) &
                            (old_bubble_table_temp['azumith'] > np.pi/3)   &
                            (old_bubble_table_temp['distance']< BUBBLE_MAXIMUM_MOVEMENT)]

        if not satisfied_bubbles.empty:

            # Satisfied bubbles are finded, find the nearest one
            old_bubble_id = satisfied_bubbles['distance'].idxmin()

            if old_bubble_table['is_counted'].at[old_bubble_id]:
                # If the bubble had been counted, update the is_counted status
                new_bubble['is_counted'] = True

            else:
                # If the bubble hasn't been counted, 
                global BUBBLE_COUNTER
                BUBBLE_COUNTER += 1
                new_bubble['is_counted'] = True
                old_bubble_table['is_counted'].at[old_bubble_id] = True

        else:
            # No satisfied previous bubble finded, take it as new uncounted detected bubble
            new_bubble['is_counted'] = False


def bubble_detection(output, frame, DETECTION_BOX):
    new_detected_bubble = pd.DataFrame(
        columns=['centroid', 'is_counted', 'stat'])
    num_labels, labels, stats, centroids = output

    for label in range(num_labels):
        this_stat = stats[label]
        this_centroid = centroids[label]

        if bubble_detector(this_stat, this_centroid, DETECTION_BOX):

            new_detected_bubble = new_detected_bubble.append(
                {'centroid': this_centroid, 'is_counted': False, 'stat': this_stat},
                ignore_index=True
            )

            cv2.rectangle(
                frame, (this_stat[0:2]), (this_stat[0:2]+this_stat[2:4]), (0, 0, 255), 2)

    return new_detected_bubble


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

    # Parameter Settings
    FPS = vc.get(cv2.CAP_PROP_FPS)
    HEIGHT = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    WIDTH = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    LENGTH = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    DETECTION_BOX = np.array([0, 430, WIDTH, 70])     # box of detection area

    old_detected_bubble = pd.DataFrame(
        columns=['centroid', 'is_counted', 'stat'])

    # Read first frame and use that as the background
    ret, bg_frame = vc.read()
    bg_edge = process_frame(bg_frame)

    while ret:
        ret, frame = vc.read()

        # Get bubble binary image
        edge_frame = process_frame(frame)
        bin_bubble_frame = get_bin_image(edge_frame, bg_edge)

        # Perform connected component analysis
        output = cv2.connectedComponentsWithStats(
            bin_bubble_frame, connectivity=8, ltype=cv2.CV_32S)

        # Bubble detection
        new_detected_bubble = bubble_detection(output, frame, DETECTION_BOX)

        # Constraints tracking
        update_bubble(new_detected_bubble, old_detected_bubble)
        old_detected_bubble = new_detected_bubble

        # Draw detection area
        cv2.rectangle(
            frame, DETECTION_BOX[0:2], DETECTION_BOX[0:2]+DETECTION_BOX[2:4], (0, 0, 255), 1)

        # Display text information
        cv2.rectangle(frame, (10, 2), (200, 20), (255, 255, 255), -1)
        cv2.putText(frame, ('COUNTER '+str(BUBBLE_COUNTER)), (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv2.imshow('Frame', frame)
        cv2.imshow('Bubble Edge', bin_bubble_frame)

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break


if __name__ == '__main__':
    main()
