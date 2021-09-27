import numpy as np
import pandas as pd
import cv2

from utilities import process_frame, get_bin_image, bubble_detector

# Global constants / Hyperparameters
BUBBLE_MAXIMUM_MOVEMENT = 30
WIDTH = 1280
DETECTION_BOX = np.array([0, 430, WIDTH, 70])       # box of detection area
CM_PX_RATIO = 9e-3                                  # ratio for convert px to cm
FIRST_PERIOD_END_TIME = 122.69                      # end time of first gas injection
SHOW_VIDEO = True                                   # show video while processing?

# Statistics
BUBBLE_COUNTER = 0
TOTAL_VOLUME = 0                                    # unit: mL
RECORDED_BUBBLE = pd.DataFrame(columns=['radius', 'movement'])


def estimate_radius(stat, shape=None):
    """
    Estimate radius of bubble according to stat returned form connected component analysis
    Bubble shape maybe used further 

    :stat: [left_top_x, left_top_y, width, height, area]
    :shape: undefined
    """

    approx_radius_px = np.sqrt(np.mean(stat[4]) / np.pi)
    approx_radius_cm = approx_radius_px * CM_PX_RATIO
    return approx_radius_cm


def update_bubble(new_bubble_table, old_bubble_table):
    """
    Find relation between old bubble table and new bubble table according to some constraints

    bubble_table <pd.DataFrame>
        'centroid' | 'is_counted' | 'stat' 
    0    (x, y)       True/False     [stat] 

    :new_bubble_table: new detected bubbles
    :old_bubble_table: old detected bubbles
    """
    if old_bubble_table.empty:
        # If there is no bubbles detected before, all new detected bubbles are uncounted
        new_bubble_table['is_counted'] = False
        return

    for new_id, new_bubble in new_bubble_table.iterrows():

        old_bubble_table_temp = old_bubble_table.copy()

        # Compute the azimuth between any old bubble and new bubble
        relative_coordinate = np.stack(
            old_bubble_table['centroid'].to_numpy()) - new_bubble['centroid']
        old_bubble_table_temp['azimuth'] = np.arctan2(
            relative_coordinate[:, 1], relative_coordinate[:, 0])
        old_bubble_table_temp['distance'] = np.linalg.norm(
            relative_coordinate, axis=1)

        # Find satisfied bubbles
        satisfied_bubbles = old_bubble_table_temp[
            (old_bubble_table_temp['azimuth'] < np.pi/3*2) &
            (old_bubble_table_temp['azimuth'] > np.pi/3) &
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
                r_hat = estimate_radius(new_bubble['stat'])
                TOTAL_VOLUME += 4/3 * np.pi * pow(r_hat, 3)

                global RECORDED_BUBBLE
                # bubble movement: (-up/+down, -left/+right)
                bubble_movement = new_bubble['centroid'] - old_bubble_table['centroid'].at[old_bubble_id]
                RECORDED_BUBBLE = RECORDED_BUBBLE.append(
                    {'radius': r_hat, 'movement': bubble_movement},
                    ignore_index=True
                )

                new_bubble['is_counted'] = True
                old_bubble_table['is_counted'].at[old_bubble_id] = True

        else:
            # No satisfied previous bubble finded, take it as new uncounted detected bubble
            new_bubble['is_counted'] = False


def bubble_detection(cca_output):
    """
    Detect bubbles from the output of connected component analysis
    The detection criterion is determined by bubble_detector

    :cca_output: output of connected component analysis
    """

    new_detected_bubble = pd.DataFrame(
        columns=['centroid', 'is_counted', 'stat'])
    num_labels, labels, stats, centroids = cca_output

    for label in range(num_labels):
        this_stat = stats[label]
        this_centroid = centroids[label]

        if bubble_detector(this_stat, this_centroid, DETECTION_BOX):

            new_detected_bubble = new_detected_bubble.append(
                {'centroid': this_centroid, 'is_counted': False, 'stat': this_stat},
                ignore_index=True
            )

    return new_detected_bubble


def add_detected_bubble_box(frame, bubble_table, current_time):
    """
    Add boxes that surround bubbles to the frame according to bubble_table

    :frame: frame to be processed
    :bubble_table: detected bubble table
    """
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

    cv2.putText(frame, ('TIME: ' + '{:.2f}'.format(current_time)+' s'),
                (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return frame


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


def save_results(frame):
    cv2.imwrite(('ratio-'+str(CM_PX_RATIO)+'.png'),
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    RECORDED_BUBBLE.to_csv('result.csv')


def main():

    ret, vc, bg_edge = initialization('output.mp4')

    # Initialize detected bubble table
    old_detected_bubble = pd.DataFrame(
        columns=['centroid', 'is_counted', 'stat'])

    while ret:
        # Read frame and get current time
        ret, frame = vc.read()
        current_time = vc.get(cv2.CAP_PROP_POS_FRAMES) / \
                vc.get(cv2.CAP_PROP_FPS)

        # Get bubble binary image
        edge_frame = process_frame(frame)
        bin_bubble_frame = get_bin_image(edge_frame, bg_edge)

        # Perform connected component analysis
        output = cv2.connectedComponentsWithStats(
            bin_bubble_frame, connectivity=8, ltype=cv2.CV_32S)

        # Bubble detection
        new_detected_bubble = bubble_detection(output)

        # Constraints tracking
        update_bubble(new_detected_bubble, old_detected_bubble)
        
        # Update detected bubble table
        old_detected_bubble = new_detected_bubble

        # Show result
        if SHOW_VIDEO:
            show_frame = add_detected_bubble_box(frame, old_detected_bubble, current_time)
            cv2.imshow('Result', show_frame)
        
        # Keyboard break
        keyboard = cv2.waitKey(1)
        if keyboard == 'q' or keyboard == 27:
            break

        # End processing and save result
        if current_time > FIRST_PERIOD_END_TIME:
            frame = add_detected_bubble_box(frame, old_detected_bubble, current_time)
            save_results(frame)
            break


if __name__ == '__main__':
    main()
