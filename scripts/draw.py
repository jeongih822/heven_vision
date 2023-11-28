import cv2
import numpy as np

# Draw Line
img = np.full((512, 512, 3), 255, np.uint8)

cv2.line(img, (0,0), (255, 255), (255, 0, 0), 3)

cv2.imshow('Line', img)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()

# Draw Rectangle
img2 = np.full((512, 512, 3), 255, np.uint8)

cv2.rectangle(img2, (20, 20), (255, 255), (255, 0, 0), 3)

cv2.imshow('Rectangle', img2)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
    
# Put Text
img3 = np.full((512, 512, 3), 255, np.uint8)

cv2.putText(img3, 'HEVEN', (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0))

cv2.imshow('Text', img3)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()