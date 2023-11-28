import cv2

src = cv2.imread('../image/noise.png', cv2.IMREAD_GRAYSCALE)

cv2.imshow('src', src)

dst = cv2.medianBlur(src, 5)
# 다양한 크기의 커널을 사용한 평균값 필터링 예제
for ksize in (3, 5, 7):
    dst = cv2.medianBlur(src, ksize)
    
    desc = 'Median : {}'.format(ksize)
    cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 1, cv2.LINE_AA)
    
    cv2.imshow('dst', dst)
    cv2.waitKey()


cv2.destroyAllWindows()