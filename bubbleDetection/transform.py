from skimage import exposure
import numpy as np
import cv2
import torchvision

def bubble_resize(bubimg, size=64): 
    transforms = [torchvision.transforms.ToTensor()] 
    width = height = size
    scale = min(width / bubimg.shape[1], height / bubimg.shape[0]) 
    new_width, new_height = int(bubimg.shape[1] * scale), int(bubimg.shape[0] * scale)
    diff_width, diff_height = width - new_width, height - new_height
    resize = torchvision.transforms.Resize(size=(new_height, new_width))
    pad = torchvision.transforms.Pad(
        padding=(
            diff_width // 2,
            diff_height // 2,
            diff_width // 2 + diff_width % 2,
            diff_height // 2 + diff_height % 2,
        )
    )
    transforms = transforms +  [resize, pad] 
    transformation = torchvision.transforms.Compose(transforms)
    return transformation(bubimg)


def transform_crop(img, crop=(1820, 0, 2420, 400)):
    # image crop
    img = img[crop[1] : crop[3], crop[0]: crop[2]]
    return img

def transform_equalization(img, crop=(1820, 0, 2420, 400)): 
    # Equalization
    img = img[crop[1] : crop[3], crop[0]: crop[2]]
    img = exposure.equalize_hist(img)
    return img

def transform_binimg(img, crop = (1820, 0, 2420, 400), threshold=0.8): 
    
    # image crop
    img = img[crop[1] : crop[3], crop[0]: crop[2]]

    # Equalization
    img = np.array(img)
    img = exposure.equalize_hist(img)

    # # convert to gray 
    img = np.dot(img, [0.2989, 0.5870, 0.1140])

    # # thresholding
    img[img < threshold] = 0 
    img[img >= threshold] = 255

    return img.astype(np.uint8)

def transform_cc(img, crop = (1820, 0, 2420, 400), bin_threshold=0.8, cc_area_th=100):
        # image crop
    img = img[crop[1] : crop[3], crop[0]: crop[2], :]
    h, w = img.shape[:2]

    # Equalization
    img = np.array(img)
    img = exposure.equalize_hist(img)

    #  convert to gray 
    img = np.dot(img, [0.2989, 0.5870, 0.1140])

    # thresholding
    img[img < bin_threshold] = 0 
    img[img >= bin_threshold] = 255
    img = img.astype(np.uint8)

    # keep satisfied connected component 
    cca_output = cv2.connectedComponentsWithStats(img, connectivity=8)
    cc_label = set() 
    num_labels, labels, stats, centroids = cca_output
    for idx, (_, _, _, _, area) in enumerate(stats[1:], start=1): 
        if area  > cc_area_th: 
            cc_label.add(idx)  
    for i in range(h): 
        for j in range(w): 
            if labels[i][j] in cc_label: 
                img[i][j] = 255
            else: 
                img[i][j] = 0 

    # flood filling 
    flood_img = img.copy()
    mask = np.ones((h+2, w+2), np.uint8) * 0
    cv2.floodFill(flood_img, mask, (0, 0), 255)
    floodfill_inv = cv2.bitwise_not(flood_img)
    img |= floodfill_inv

    return img