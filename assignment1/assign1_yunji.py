import cv2
import numpy as np

def draw_checkerboard(img, N, M):
    width = img.shape[1]//N
    height = img.shape[0]//N

    for i in range(N):
        for j in range(N):
            x = j * width
            y = i * height
            
            if (i//M + j//M) % 2 == 0:
                img[y:y+height, x:x+width] = 0


if __name__ == "__main__":
    
    N = int(input("Enter the number of N : "))
    
    M = int(input("Enter the number of M : "))
    
    img = np.full((N, N, 1), 255, np.uint8)
    
    draw_checkerboard(img, N, M)

    canny_img = cv2.Canny(img, 100, 255)

    lines = cv2.HoughLinesP(canny_img, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    final_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if lines is not None:
        for i in range(lines.shape[0]):
            pt1 = (lines[i][0][0], lines[i][0][1])
            pt2 = (lines[i][0][2], lines[i][0][3])
            
            cv2.line(final_img, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Result", final_img)
    
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()

