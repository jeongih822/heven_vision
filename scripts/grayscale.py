import cv2

img = cv2.imread('../image/cat.jpeg')

converted_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('GrayScale', converted_img1)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()

converted_img2 = cv2.imread('../image/cat.jpeg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('GrayScale', converted_img2)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()