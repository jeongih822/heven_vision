import cv2
import numpy as np

def draw_checkerboard(img, N, M):
    # Create the checkerboard pattern
    for i in range(0, N, M*2):
        for j in range(0, N, M*2):
            img[i:i+M, j:j+M] = 0  # Set black squares

            if i+M < N:  # Avoid out-of-bounds error
                img[i+M:i+M*2, j+M:j+M*2] = 0  # Set black squares in alternate rows
    

if __name__ == "__main__":
    
    N = int(input("Enter the number of N : "))
    
    M = int(input("Enter the number of M : "))
    
    img = np.full((N, N, 1), 255, np.uint8)
    
    draw_checkerboard(img, N, M)

    #edge detection -- canny
    canny_img = cv2.Canny(img, 100, 255)

    #lane detection and draw
    threshold=100
    
    lines = cv2.HoughLinesP(canny_img, 1, np.pi/180, threshold, minLineLength=100, maxLineGap=5)

    hough_img = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)
    text = 'houghP'
        
    cv2.putText(hough_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
    if lines is not None:
        for i in range(lines.shape[0]):
            pt1 = (lines[i][0][0], lines[i][0][1])
            pt2 = (lines[i][0][2], lines[i][0][3])
            
            cv2.line
            (hough_img, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    
    
    cv2.imshow("Result", img)
    cv2.imshow('Canny', canny_img)
    cv2.imshow('Hough', hough_img)
    #cv2.imshow("Result", final_img)
    
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()


        