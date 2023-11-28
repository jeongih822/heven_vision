import cv2

img = cv2.imread('../image/cat.jpeg')

print("Type : ", type(img))

print("Dimension : ", img.shape)

print("Pixel value of (0, 0) : ", img[0][0])

cv2.imshow('Display', img)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()