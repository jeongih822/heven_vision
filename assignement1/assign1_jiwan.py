import cv2
import numpy as np

def draw_checkerboard(img, N, M):
    num_squares = N // M
    for row in range(num_squares):
        for col in range(num_squares):
            if (row + col) % 2 == 0:
                top_left_y = row * M
                top_left_x = col * M
                img[top_left_y:top_left_y + M, top_left_x:top_left_x + M] = 0


if __name__ == "__main__":
    
    N = int(input("Enter the number of N : "))
    
    M = int(input("Enter the number of M : "))
    
    img = np.full((N, N, 1), 255, np.uint8)
    
    draw_checkerboard(img, N, M)
    canny_img = cv2.Canny(img, 70, 210)
    lines = cv2.HoughLinesP(canny_img, 1, np.pi/180, threshold=200, minLineLength=100, maxLineGap=5)
    hough_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for i in range(lines.shape[0]):
            pt1 = (lines[i][0][0], lines[i][0][1])
            pt2 = (lines[i][0][2], lines[i][0][3])
            cv2.line(hough_img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("Result", hough_img)   
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
