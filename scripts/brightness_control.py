import cv2
import numpy as np

img = cv2.imread('../image/cat.jpeg', cv2.IMREAD_GRAYSCALE)

bright_img = cv2.add(img, 100)

dark_img = cv2.add(img, -100)

cv2.imshow('Original', img)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()

cv2.imshow('Bright', bright_img)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
    
cv2.imshow('Dark', dark_img)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
