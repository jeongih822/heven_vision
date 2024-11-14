#! /usr/bin/env python3

import rospy
import cv2

from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import numpy as np
import time

class GreenTrigger():
    def __init__(self):
        
        self.bridge = CvBridge()
        self.image = None

        self.index = 0
        self.last_index_time = 0
        self.running_time = 4

        rospy.init_node('CvBride_node', anonymous=False)
        rospy.Subscriber('/main_camera/image_raw/compressed', CompressedImage, self.camera_callback)
        
        
    def camera_callback(self, data):
        self.image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        self.main()
        cv2.imshow("Display", self.image)

    def color_filter(self, image):
        """
            HLS 필터 사용
            
            lower & upper : 흰색으로 판단할 minimum pixel 값
            white_mask : lower과 upper 사이의 값만 남긴 mask
            masked : cv2.bitwise_and() 함수를 통해 흰색인 부분 제외하고는 전부 검정색 처리
        """
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        lower = np.array([35, 75, 35])
        upper = np.array([75, 115, 75])

        white_mask = cv2.inRange(hls, lower, upper)
        masked = cv2.bitwise_and(image, image, mask = white_mask)

        return masked
    
    def main(self):
        if self.image is not None:
            self.image = cv2.resize(self.image, (640, 480))
            # cv2.polylines(self.image, [np.array(self.source)], True, (255, 0, 255), 2)
            
            # warpped_img, minv = self.warpping(self.image)
            # cv2.imshow("Display_warpping", warpped_img)
            blurred_img = cv2.GaussianBlur(self.image, (0, 0), 1)
            roi_img = blurred_img[200:480, 0:640]
            w_f_img = self.color_filter(roi_img)
            _gray = cv2.cvtColor(w_f_img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(_gray, 0, 100, cv2.THRESH_BINARY)
            nonzero = thresh.nonzero()

            current_time = time.time()

            if (len(nonzero[0]) > 11000) and (current_time - self.last_index_time) >= self.running_time:
                print("Green detected")
                self.index += 1
                self.last_index_time = current_time

            print("Index: ", self.index)
            
            cv2.imshow("Display_filter", thresh)
            cv2.waitKey(1)


if __name__ == "__main__":
    
    if not rospy.is_shutdown():
        GreenTrigger()
        rospy.spin()