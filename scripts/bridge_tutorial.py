#! /usr/bin/env python

import rospy
import cv2

from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge


class example():
    def __init__(self):
        
        self.bridge = CvBridge()
        
        rospy.init_node('CvBride_node', anonymous=False)
        rospy.Subscriber('/image_jpeg_2/compressed', CompressedImage, self.camera_callback)
        
        
    def camera_callback(self, data):
        self.image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        
        cv2.imshow("Display", self.image)
        cv2.waitKey(1)


if __name__ == "__main__":
    
    if not rospy.is_shutdown():
        example()
        rospy.spin()