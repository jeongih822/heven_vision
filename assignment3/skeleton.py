#! /usr/bin/env python3

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from lane_control.msg import lane_info
# lane_info
# int64 left_x
# int64 right_x
# float32 left_theta
# float32 right_theta
from math import *

class lane_detect():
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('lane_detection_node', anonymous=False)
        rospy.Subscriber('/image_jpeg_2/compressed', CompressedImage, self.camera_callback)
        self.pub = rospy.Publisher("/lane_result", lane_info, queue_size=1)

    
    def camera_callback(self, data):
        """
        code here
        """
        self.pub.publish(self.lane_detect())
    
    """
    lane detection preprocessing code here
    """
    
    def main(self):
        """
        code here(lane_detection)
        """
        pub_msg = lane_info()
        """
        code here(publish)
        """

        return pub_msg


if __name__ == "__main__":

    if not rospy.is_shutdown():
        lane_detect()
        rospy.spin()