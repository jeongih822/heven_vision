import cv2
import numpy as np

def draw_checkerboard(img, N, M):
    """
    Please write a code here
    """
    # draw checkboard
    for i in range(0, N, 2*M):
        for j in range(0, N, 2*M):
            img[i:i+M, j:j+M] = 0
    for i in range(M, N, 2*M):
        for j in range(M, N, 2*M):
            img[i:i+M, j:j+M] = 0
    
    # edge detection
    edged_img = cv2.Canny(img, 70, 120)
    cv2.imshow('Canny Edge', edged_img)
    
    lines = cv2.HoughLinesP(edged_img, 1, np.pi/180, 0, minLineLength=100, maxLineGap=5)
    hough_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if lines is not None:
        for i in range(lines.shape[0]):
            pt1 = (lines[i][0][0], lines[i][0][1])
            pt2 = (lines[i][0][2], lines[i][0][3])
            cv2.line(hough_img, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Hough Edge', hough_img)


if __name__ == "__main__":
    
    N = int(input("Enter the number of N : "))
    
    M = int(input("Enter the number of M : "))
    
    img = np.full((N, N, 1), 255, np.uint8)
    
    draw_checkerboard(img, N, M)
    
    cv2.imshow("Result", img)
    
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()