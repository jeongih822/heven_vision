import cv2
import numpy as np

img = cv2.imread('../image/lane.jpg')
h, w, c = img.shape

pts1 = np.float32([[350,40],[600,40],[735,500]]) # 입력 영상의 세 점의 위치
pts2 = np.float32([[0,0],[960,0],[960,500]])  # 변환 후 결과 영상의 세 점이 될 위치

M = cv2.getAffineTransform(pts1, pts2)  # 어파인 변환 행렬 구함

dst = cv2.warpAffine(img, M, (w, h))

cv2.imshow('img', img)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()