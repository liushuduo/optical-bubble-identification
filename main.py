import numpy as np
import pandas as pd
import cv2

from utilities import process_frame, get_bin_image, bubble_detector, shape2volume

# Global constants
BUBBLE_MAXIMUM_MOVEMENT = 30
WIDTH = 1280
DETECTION_BOX = np.array([0, 430, WIDTH, 70])       # box of detection area
CM_PX_RATIO = 9e-3                                  # ratio for convert px to cm
FIRST_PERIOD_END_TIME = 122.69                      # end time of first gas injection

# Statistics
BUBBLE_COUNTER = 0
TOTAL_VOLUME = 0                                    # unit: mL


def update_bubble(new_bubble_table, old_bubble_table):

    if old_bubble_table.empty:
        # If there is no bubbles detected before, all new detected bubbles are uncounted
        new_bubble_table['is_counted'] = False
        return

    for new_id, new_bubble in new_bubble_table.iterrows():

        old_bubble_table_temp = old_bubble_table.copy()

        # Compute the azumith between any old bubble and new bubble
        relative_coordinate = np.stack(
            old_bubble_table['centroid'].to_numpy()) - new_bubble['centroid']
        old_bubble_table_temp['azumith'] = np.arctan2(
            relative_coordinate[:, 1], relative_coordinate[:, 0])
        old_bubble_table_temp['distance'] = np.linalg.norm(
            relative_coordinate, axis=1)

        # Find satisfied bubbles
        satisfied_bubbles = old_bubble_table_temp[
            (old_bubble_table_temp['azumith'] < np.pi/3*2) &
            (old_bubble_table_temp['azumith'] > np.pi/3) &
            (old_bubble_table_temp['distance'] < BUBBLE_MAXIMUM_MOVEMENT)]

        if not satisfied_bubbles.empty:

            # Satisfied bubbles are finded, find the nearest one
            old_bubble_id = satisfied_bubbles['distance'].idxmin()

            if old_bubble_table['is_counted'].at[old_bubble_id]:
                # If the bubble had been counted, update the is_counted status
                new_bubble['is_counted'] = True

            else:
                # If the bubble hasn't been counted, count the bubble and estimate its volumn
                global BUBBLE_COUNTER
                BUBBLE_COUNTER += 1

                global TOTAL_VOLUME
                TOTAL_VOLUME += shape2volume(new_bubble['stat'], CM_PX_RATIO)

                new_bubble['is_counted'] = True
                old_bubble_table['is_counted'].at[old_bubble_id] = True

        else:
            # No satisfied previous bubble finded, take it as new uncounted detected bubble
            new_bubble['is_counted'] = False


def bubble_detection(output, frame):
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

    return new_detected_bubble


def show_detected_bubble(frame, bubble_table, vc):

    # Draw bubble boxes
    for idx, bubble in bubble_table.iterrows():
        if bubble['is_counted']:
            cv2.rectangle(
                frame, (bubble['stat'][0:2]), (bubble['stat'][0:2]+bubble['stat'][2:4]), (0, 0, 255), 1)
        else:
            cv2.rectangle(
                frame, (bubble['stat'][0:2]), (bubble['stat'][0:2]+bubble['stat'][2:4]), (0, 0, 255), 1)

    # Draw detection area
    cv2.rectangle(
        frame, DETECTION_BOX[0:2], DETECTION_BOX[0:2]+DETECTION_BOX[2:4], (0, 0, 255), 1)

    # Display text information
    cv2.rectangle(frame, (10, 2), (200, 60), (255, 255, 255), -1)

    cv2.putText(frame, ('COUNTER: '+str(BUBBLE_COUNTER)), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.putText(frame, ('VOLUME: '+'{:.5f}'.format(TOTAL_VOLUME) + ' mL'), (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    current_time = vc.get(cv2.CAP_PROP_POS_FRAMES)/vc.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, ('TIME: ' + '{:.2f}'.format(current_time)+' s'),
                (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    if FIRST_PERIOD_END_TIME <= current_time and current_time < FIRST_PERIOD_END_TIME+1/vc.get(cv2.CAP_PROP_FPS):
        cv2.imwrite('bubble-detection.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    cv2.imshow('Frame', frame)


def initialization(filename):

    vc = cv2.VideoCapture(filename)
    if vc.isOpened():
        print("Video opened!")
    else:
        print("Video open error!")

    # Read first frame and use that as the background
    ret, bg_frame = vc.read()
    bg_edge = process_frame(bg_frame)

    return ret, vc, bg_edge


def main():

    ret, vc, bg_edge = initialization('output.mp4')

    # Initialize detected bubble table
    old_detected_bubble = pd.DataFrame(
        columns=['centroid', 'is_counted', 'stat'])

    while ret:
        # Read frame
        ret, frame = vc.read()

        # Get bubble binary image
        edge_frame = process_frame(frame)
        bin_bubble_frame = get_bin_image(edge_frame, bg_edge)

        # Perform connected component analysis
        output = cv2.connectedComponentsWithStats(
            bin_bubble_frame, connectivity=8, ltype=cv2.CV_32S)

        # Bubble detection
        new_detected_bubble = bubble_detection(output, frame)

        # Constraints tracking
        update_bubble(new_detected_bubble, old_detected_bubble)

        # Show result
        show_detected_bubble(frame, new_detected_bubble, vc)

        # Update detected bubble table
        old_detected_bubble = new_detected_bubble

        keyboard = cv2.waitKey(1)
        if keyboard == 'q' or keyboard == 27:
            break


if __name__ == '__main__':
    main()
