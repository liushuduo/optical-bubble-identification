import numpy as np
import matplotlib.pyplot as plt
import cv2


def enhance_brightness(image, alpha=1, beta=0):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    v_result = np.clip(cv2.add(np.round(alpha*v), beta), 0, 255)
    img_result = np.uint8( cv2.merge((h,s,v_result)) )
    img_result = cv2.cvtColor(img_result,cv2.COLOR_HSV2BGR)
    return img_result


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def enhance_image(frame):
    brighter_frame = enhance_brightness(frame, beta=20)
    sharpened_frame = unsharp_mask(brighter_frame)
    return sharpened_frame


filename = './bubble_frames/frame-1.png'
bubble_frame_bgr = cv2.imread(filename)
cv2.imwrite('./bubble_frames/raw.png', bubble_frame_bgr)

# Sharpen frame
sharpened_frame = unsharp_mask(bubble_frame_bgr)
cv2.imwrite('./bubble_frames/sharpened-frame.png', sharpened_frame)

# Enhance frame brightness
brighter_frame = enhance_brightness(bubble_frame_bgr, alpha=1.1, beta=30)
cv2.imwrite('./bubble_frames/brighter-frame.png', brighter_frame)

# Enhance frame
enhanced_frame = enhance_image(brighter_frame)
cv2.imwrite('./bubble_frames/enhanced-frame.png', enhanced_frame)