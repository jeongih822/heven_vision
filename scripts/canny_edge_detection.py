import cv2

img = cv2.imread('../image/lane.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('Original', img)

for (i, j) in [(70, 210), (100, 255), (220, 255)]:
    
    canny_img = cv2.Canny(img, i, j)

    cv2.imshow('Canny', canny_img)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()