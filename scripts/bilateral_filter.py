import cv2

src = cv2.imread('../image/test.png')

dst = cv2.bilateralFilter(src, -1, 10, 10)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()

cv2.destroyAllWindows()