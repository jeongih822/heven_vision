import cv2

src = cv2.imread('../image/lane.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('src', src)

for sigma in range(1, 4):
    dst = cv2.GaussianBlur(src, (0, 0), sigma)
    
    desc = 'sigma = {}'.format(sigma)
    cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
    
    cv2.imshow('dst', dst)
    cv2.waitKey()
    
cv2.destroyAllWindows