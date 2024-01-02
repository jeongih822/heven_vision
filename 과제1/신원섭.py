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
                img[start_row:end_row, start_col:end_col]=0 #검정
            else:
                img[start_row:end_row, start_col:end_col]=255 #흰색

            
            

if __name__ == "__main__":
    
    N = int(input("Enter the number of N : "))
    
    M = int(input("Enter the number of M : "))
    
    img = np.full((N, N, 1), 255, np.uint8)
    
    draw_checkerboard(img, N, M)
    
    cv2.imshow("Result", img)
    
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()