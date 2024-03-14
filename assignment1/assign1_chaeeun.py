import cv2
import numpy as np

def draw_checkerboard(img, N, M):
    numSquare = int(N / M)
    nowColor = 0
    for row in range(numSquare):
        for col in range(numSquare):
            if((row + col) % 2 == 0):
                img[M*row:M*(row+1), M*col:M*(col+1)] = nowColor
        


if __name__ == "__main__":
    
    N = int(input("Enter the number of N : "))
    
    M = int(input("Enter the number of M : "))
    
    img = np.full((N, N, 1), 255, np.uint8)    

    draw_checkerboard(img, N, M)

    canny_img = cv2.Canny(img, 70, 210)    
    lines = cv2.HoughLinesP(canny_img, 1, np.pi/180, 0, minLineLength=100, maxLineGap=5)
    hough_img = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)
    
    if lines is not None:
        for i in range(lines.shape[0]):
            pt1 = (lines[i][0][0], lines[i][0][1])
            pt2 = (lines[i][0][2], lines[i][0][3])
            cv2.line(hough_img,pt1,pt2,(0,0,255),2,cv2.LINE_AA)
    
    cv2.imshow("RESULT",hough_img)

    if cv2.waitKey(0) == 27:
        cv2.destoryAllWindows()


        