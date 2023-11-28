import cv2

img = cv2.imread('../image/lane2.jpeg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('Gray', img)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()

images = []

ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

images.append(thresh1)
images.append(thresh2)
images.append(thresh3)
images.append(thresh4)
images.append(thresh5)

for image in images:
    cv2.imshow('Display', cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()