import numpy as np
import cv2


def is_inarea(centroid, DETECTION_BOX):
    """
    Detect if centroid is in the area of DETECTION_BOX
    """
    if centroid - DETECTION_BOX[0:2] < DETECTION_BOX[2:4]:
        return True 
    

def is_inshape(stats):
    """
    Detect if the connected component is in shape of bubble

    :stats: cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
            cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.
            cv2.CC_STAT_WIDTH The horizontal size of the bounding box
            cv2.CC_STAT_HEIGHT The vertical size of the bounding box
            cv2.CC_STAT_AREA The total area (in pixels) of the connected component
    """
    if stats[4] < 40:
        # component with area less than 40 pixel^2 is not considered as bubble
        return False

    elif stats[4] > 5000:
        #  component with area larger than 5000 pixel^2 is not considered
        return False

    else:
        return True

def bubble_detector(stats, centroid, DETECTION_BOX):
    """
    A simple bubble detector. If a connected component enters the 
    detection area and its area is greater than TH, it would be 
    detected.

    :stats:         stats returned from cv2.connectedCom..
    :centroid:      position of bubble center
    :th:            bubble area threshold
    """
    return (is_inarea(centroid, DETECTION_BOX) and is_inshape(stats))

def process_frame(frame, sharpen_kernel=np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])):
    """ 
    Detect the edge of frame

    :frame:             a single frame
    :sharpen_kernel:    kernel of filter for sharpening image
    :return:            detected edge of frame
    """

    # Convert to gray
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Sharpen frame
    sharpened_frame = cv2.filter2D(gray_frame, -1, sharpen_kernel)

    # Edge detection using absolute Sobel detector
    dx = cv2.Sobel(sharpened_frame, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(sharpened_frame, cv2.CV_64F, 0, 1)
    abs_dx = cv2.convertScaleAbs(dx)
    abs_dy = cv2.convertScaleAbs(dy)
    edge_frame = cv2.addWeighted(abs_dx, 0.5, abs_dy, 0.5, 0)

    return edge_frame


def get_bin_image(edge_frame, bg_edge, threshold=70):
    """
    Get the binary image of bubbles 

    :edge_frame:    edge detected frame
    :bg_edge:       background edge
    :threshold:     threshold for image threshold
    """

    height, width = edge_frame.shape
    bubble_frame = cv2.subtract(edge_frame, bg_edge)

    # Guassian blur to smooth image
    bubble_frame_blur = cv2.GaussianBlur(bubble_frame, (3, 3), 0)

    # Image threshold
    ret, bin_bubble_frame = cv2.threshold(
        bubble_frame_blur, threshold, 255, cv2.THRESH_BINARY)

    # Flood filling the holes of bubbles
    bubble_frame_floodfill = bin_bubble_frame.copy()

    # Mask size needs to be 2 pixels larger than the image.
    mask = np.ones((height+2, width+2), np.uint8) * 0
    cv2.floodFill(bubble_frame_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    bubble_frame_floodfill_inv = cv2.bitwise_not(bubble_frame_floodfill)
    bubble_frame_result = bin_bubble_frame | bubble_frame_floodfill_inv

    return bubble_frame_result


def new_method(frame, bg_frame, sharpen_kernel=np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])):
    """
    Test method: perform background subtraction first, then perform edge detection.
    This method has been proven poor.

    :frame:             input frame
    :bg_frame:          input background frame
    :sharpen_kernel:    kernel to perform image sharpening
    """

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    bg_frame = cv2.cvtColor(bg_frame, cv2.COLOR_RGB2GRAY)
    height, width = frame.shape

    bg_frame_blur = cv2.GaussianBlur(bg_frame, (3, 3), 0)
    bubble_frame = cv2.subtract(frame, bg_frame_blur)

    # Sharpen frame
    sharpened_frame = cv2.filter2D(bubble_frame, -1, sharpen_kernel)

    # Edge detection using absolute Sobel detector
    dx = cv2.Sobel(sharpened_frame, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(sharpened_frame, cv2.CV_64F, 0, 1)
    abs_dx = cv2.convertScaleAbs(dx)
    abs_dy = cv2.convertScaleAbs(dy)
    edge_frame = cv2.addWeighted(abs_dx, 0.5, abs_dy, 0.5, 0)

    # Guassian blur to smooth image
    bubble_frame_blur = cv2.GaussianBlur(edge_frame, (3, 3), 0)

    # Image threshold
    ret, bin_bubble_frame = cv2.threshold(
        bubble_frame_blur, 70, 255, cv2.THRESH_BINARY)

    # Flood filling the holes of bubbles
    bubble_frame_floodfill = bin_bubble_frame.copy()

    # Mask size needs to be 2 pixels larger than the image.
    mask = np.ones((height+2, width+2), np.uint8) * 0
    cv2.floodFill(bubble_frame_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    bubble_frame_floodfill_inv = cv2.bitwise_not(bubble_frame_floodfill)
    bubble_frame_result = bin_bubble_frame | bubble_frame_floodfill_inv

    return bubble_frame_result


if __name__ == '__main__':

    vc = cv2.VideoCapture("output.mp4")
    ret, bg_frame = vc.read()

    while True:
        ret, frame = vc.read()
        if frame is None:
            break

        bin_bubble_frame = new_method(frame, bg_frame)

        # # Get bubble binary image
        # edge_frame = process_frame(frame)
        # bin_bubble_frame = get_bin_image(edge_frame, bg_edge)

        # output = cv2.connectedComponentsWithStats(
        #     bin_bubble_frame, connectivity=8, ltype=cv2.CV_32S)
        # num_labels, labels, stats, centroids = output

        # for this_stat in stats[1:]:
        #     if this_stat[4] > 40:
        #         cv2.rectangle(frame, (this_stat[0:2]), (this_stat[0:2]+this_stat[2:4]),
        #                       (0, 0, 255), 2)

        cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putText(frame, str(vc.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv2.imshow('Frame', frame)
        cv2.imshow('Bubble Edge', bin_bubble_frame)

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
