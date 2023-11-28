import cv2
import numpy as np

img = cv2.imread('../image/lane.jpg', cv2.IMREAD_GRAYSCALE)

blurred_img = cv2.GaussianBlur(img, (0, 0), 2)

canny_img = cv2.Canny(blurred_img, 70, 210)

for threshold in [0, 50, 100, 160, 200]:

    lines = cv2.HoughLinesP(canny_img, 1, np.pi/180, threshold, minLineLength=100, maxLineGap=5)

    hough_img = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)
    text = 'threshold = {}'.format(threshold)
    
    cv2.putText(hough_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
    if lines is not None:
        for i in range(lines.shape[0]):
            pt1 = (lines[i][0][0], lines[i][0][1])
            pt2 = (lines[i][0][2], lines[i][0][3])
            
            cv2.line(hough_img, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Hough', hough_img)
    cv2.waitKey(0)

cv2.destroyAllWindows()