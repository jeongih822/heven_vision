import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../image/lane.jpg')
# [x,y] 좌표점을 4x2의 행렬로 작성
# 좌표점은 좌상->좌하->우상->우하
pts1 = np.float32([[350,40], [220, 500], [600, 40], [735,500]])

# 좌표의 이동점
pts2 = np.float32([[0,0], [0,460], [250,0], [250,460]])

# pts1의 좌표에 표시. perspective 변환 후 이동 점 확인.
cv2.circle(img, (350, 40), 20, (255,0,0),-1)
cv2.circle(img, (220, 500), 20, (0,255,0),-1)
cv2.circle(img, (600, 40), 20, (0,0,255),-1)
cv2.circle(img, (735, 500), 20, (0,0,0),-1)

M = cv2.getPerspectiveTransform(pts1, pts2)

dst = cv2.warpPerspective(img, M, (250, 460))

cv2.imshow('Original', img)
cv2.imshow('Transformed', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()