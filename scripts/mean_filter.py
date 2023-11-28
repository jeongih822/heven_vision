import cv2

img = cv2.imread('../image/lane.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('src', img)

# 다양한 크기의 커널을 사용한 평균값 필터링 예제
for ksize in (3, 5, 7):
    dst = cv2.blur(img, (ksize, ksize))
    
    desc = 'Mean : {}x{}'.format(ksize, ksize)
    cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
    
    cv2.imshow('dst', dst)
    cv2.waitKey()
    
cv2.destroyAllWindowns()