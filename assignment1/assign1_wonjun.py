import cv2
import numpy as np


def draw_checkerboard(img, N, M):
    """
    Please write a code here
    """
    flag = 0
    for i in range(0, N, M):
        flag = abs(flag - 1)
        for j in range(0, N, M):
            if flag == 0:
                cv2.rectangle(img, (j, i), (j + M, i + M), (0, 0, 0), -1)
                flag = 1
            else:
                cv2.rectangle(img, (j, i), (j + M, i + M), (0, 0, 0))
                flag = 0


if __name__ == "__main__":

    # N = int(input("Enter the number of N : "))
    N = 500

    # M = int(input("Enter the number of M : "))
    M = 50

    img = np.full((N, N, 1), 255, np.uint8)

    draw_checkerboard(img, N, M)
    original = img
    img = cv2.Canny(img, 220, 255)
    lines = cv2.HoughLines(img, 1, np.pi / 180, 200)

    original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for i in range(lines.shape[0]):
            rho, theta = lines[i][0][0], lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(original, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("Result", original)
    cv2.moveWindow("Result", 100, 100)

    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
