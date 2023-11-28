import cv2

img = cv2.imread('../image/lane.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('Original', img)

for sigma in range(1, 4):
    
    blurred_img = cv2.GaussianBlur(img, (0, 0), sigma)
    
    text = 'sigma = {}'.format(sigma)
    for (i, j) in [(70, 210), (100, 255), (220, 255)]:
        
        canny_img = cv2.Canny(blurred_img, i, j)

        text2 = 'threshold = {}'.format((i, j))
        cv2.putText(canny_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
        cv2.putText(canny_img, text2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
        cv2.imshow('Canny', canny_img)
        cv2.waitKey(0)
    
cv2.destroyAllWindows()