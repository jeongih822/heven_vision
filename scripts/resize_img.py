import cv2

img = cv2.imread('../image/cat.jpeg')

expand1 = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

cv2.imshow('Original', img)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()



cv2.imshow('First Resize', expand1)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()



