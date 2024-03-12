import cv2
import numpy as np


def draw_checkerboard(img, N, M):
    color = 0
    hell = 0
    for i in range(int(N/M)):
        hell = (hell+1)%2
        for j in range(int(N/M)):
            if j % 2 ==hell :
                img[i*M : (i+1)*M , j*M : (j+1)*M] = 0
            
    


if __name__ == "__main__":

    N = int(input("Enter the number of N :"))
    M = int(input("Enter the number of M : "))
    img = np.full((N,N,1),255,np.uint8)
    draw_checkerboard(img,N,M)

    """
    code
    """
    edge_img = cv2.Canny(img,150,100)

    lines = cv2.HoughLinesP(edge_img, 1, np.pi/180 , 0, minLineLength = M, maxLineGap = 5)
    
    hough_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for i in range(lines.shape[0]):
            pt1 = (lines[i][0][0], lines[i][0][1])
            pt2 = (lines[i][0][2], lines[i][0][3])
            cv2.line(hough_img,pt1,pt2,(0,0,255),2,cv2.LINE_AA)
    
    cv2.imshow("Checker Board",hough_img)

    if cv2.waitKey(0) == 27:
        cv2.destoryAllWindows()


