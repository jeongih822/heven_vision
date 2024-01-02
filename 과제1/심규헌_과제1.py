import cv2
import numpy as np

def draw_checkerboard(img, N, M):
    square_size = N // M

    for i in range(N):
        for j in range(N):
            if (i // square_size + j // square_size) % 2 == 0:
                img[i, j] = 0

if __name__ == "__main__":
    N = int(input("Enter the number of N: "))
    M = int(input("Enter the number of M: "))
    
    img = np.full((N, N, 1), 255, np.uint8)
    
    draw_checkerboard(img, N, M)
    
    cv2.imshow("Result", img)
    
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
