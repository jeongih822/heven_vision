import cv2
import numpy as np

def draw_checkerboard(img, N, M):
    square_size = N // M

    for i in range(M):
        for j in range(M):
            start_row = i * square_size
            start_col = j * square_size

            end_row = (i+1) * square_size
            end_col = (j+1) * square_size
            
            if (i+j)%2 == 0:
                img[start_row:end_row, start_col:end_col]=0 
            else:
                img[start_row:end_row, start_col:end_col]=255         

if __name__ == "__main__":
    
    N = int(input("Enter the number of N : "))
    
    M = int(input("Enter the number of M : "))
    
    img = np.full((N, N, 1), 255, np.uint8)
    
    draw_checkerboard(img, N, M)

    #edge detection
    edge_img = cv2.Canny(img, 50, 150)

    #line detection
    lines = cv2.HoughLinesP(edge_img, 1, np.pi/180, threshold=200, minLineLength=100, maxLineGap=5)

    hough_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for i in range(lines.shape[0]):
            pt1 = (lines[i][0][0], lines[i][0][1])
            pt2 = (lines[i][0][2], lines[i][0][3])

            cv2.line(hough_img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("Result", hough_img)
    
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()