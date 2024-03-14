import cv2
import numpy as np

def draw_checkerboard(img, N, M):
    num_cells = N // M
    cell_size = M
    for row in range(num_cells):
        for col in range(num_cells):
            if((row + col) % 2 == 0):
                x1 = M * row
                y1 = M * col
                img[x1:x1+cell_size, y1:y1+cell_size] = 0


if __name__ == "__main__":
    
    N = int(input("Enter the number of N : "))
    
    M = int(input("Enter the number of M : "))
    
    img = np.full((N, N, 1), 255, np.uint8)    

    draw_checkerboard(img, N, M)
    
    """
    Please write edge detection code here
    """
    canny_img = cv2.Canny(img, 50, 200)    
    lines = cv2.HoughLinesP(canny_img, 1, np.pi/180, 0, minLineLength=100, maxLineGap=5)
    final_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if lines is not None:
        for i in range(lines.shape[0]):
            pt1 = (lines[i][0][0], lines[i][0][1])
            pt2 = (lines[i][0][2], lines[i][0][3])
            cv2.line(final_img, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("Result", final_img)

    if cv2.waitKey(0) == 27:
        cv2.destoryAllWindows()
