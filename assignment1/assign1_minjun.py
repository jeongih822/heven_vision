import cv2
import numpy as np

def draw_checkboard(img, N, M):
    for i in range(N//M):
        for j in range(N//M):
            img[2*M*i:2*M*i+M, 2*M*j:2*M*j+M]=0
            img[M+2*M*i:M+2*M*i+M, M+2*M*j:2*M*j+2*M]=0 

if __name__ == "__main__":

    N = int(input("Enter the number of N : "))

    M = int(input("Enter the number of M : "))

    img = np.full((N, N, 1), 255, np.uint8)

    draw_checkboard(img, N, M)

    canny_img = cv2.Canny(img, 70, 210)
    lines = cv2.HoughLinesP(canny_img, 1, np.pi/180, 100, minLineLength=100, maxLineGap=5)
    final_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for i in range(lines.shape[0]):
            pt1 = (lines[i][0][0], lines[i][0][1])
            pt2 = (lines[i][0][2], lines[i][0][3])
            cv2.line(final_img, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Result", final_img)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()