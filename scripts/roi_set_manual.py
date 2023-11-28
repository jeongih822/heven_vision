import cv2, numpy as np

cap = cv2.VideoCapture('../video/curve_lane.mp4')

while True:
    
    ret, frame = cap.read()
    
    print(frame.shape)
    
    x,y,w,h = cv2.selectROI('img', frame, True)
    
    print("x : ", x)
    print("y : ", y)

    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
