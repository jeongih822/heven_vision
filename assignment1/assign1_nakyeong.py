import cv2
import numpy as np

def draw_checkerboard(img, N, M):
    cell_width = img.shape[1] // M
    cell_height = img.shape[0] // N
    
    for i in range(N):
        for j in range(M):
            if (i + j) % 2 == 0:
                img[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width] = 0
            else:
                img[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width] = 255

if __name__ == "__main__":
    N = int(input("Enter the number of rows (N): "))
    M = int(input("Enter the number of columns (M): "))

    img = np.full((N, M, 1), 255, np.uint8)

    draw_checkerboard(img, N, M)
    
    cv2.imshow("Result", img)

    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
