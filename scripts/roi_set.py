import cv2

img = cv2.imread('../image/cat.jpeg')

(x,y) = (100, 100)
(w,h) = (30, 20)

roi = img[y:y+h, x:x+w]

# roi
cv2.imshow('Display', roi)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()

cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow('Display', img)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()