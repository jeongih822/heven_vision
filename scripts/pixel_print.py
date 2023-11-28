import cv2

img = cv2.imread('../image/cat.jpeg')

print(img.shape)
print(img.size)

pixel = img[100, 100]

print(pixel)