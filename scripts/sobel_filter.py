import cv2

img = cv2.imread('../image/lane.jpg', cv2.IMREAD_GRAYSCALE)

x_sobel_img = cv2.Sobel(img, -1, 1, 0, ksize=3)
y_sobel_img = cv2.Sobel(img, -1, 0, 1, ksize=3)
sobel_img = cv2.Sobel(img, -1, 1, 1, ksize=3)

cv2.imshow('X', x_sobel_img)
cv2.imshow('Y', y_sobel_img)
cv2.imshow('Sobel', sobel_img)

cv2.waitKey(0)
cv2.destroyAllWindows()