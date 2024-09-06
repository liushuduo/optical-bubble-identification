import cv2

def binimg_bubble_detector(binimg, area_threshold=100):       
      
    cca_output = cv2.connectedComponentsWithStats(binimg, connectivity=8)
    ans = []
    num_labels, labels, stats, centroids = cca_output
    for x, y, w, h, area in stats[1:]: 
        if area  > area_threshold: 
            ans.append((x, y, w, h))
    return ans
