import cv2

path = '/home/yeong/catkin_ws/src/heven_vision/video/testVideo.mp4'
cap = cv2.VideoCapture(path)

while True:
    ret, frame = cap.read()
    
    cv2.imshow('Display', frame)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()