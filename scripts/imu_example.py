#!/usr/bin/env python3

import rospy
import numpy as np

from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion
from math import *

class IMU():
    def __init__(self):
        rospy.init_node('imu_sample', anonymous=True)
        self._imu_yaw = 0
        rospy.Subscriber("/imu", Imu, self._imu_callback)

    def _imu_callback(self, data=Imu):
        w = data.orientation.w
        x = data.orientation.x
        y = data.orientation.y
        z = data.orientation.z
    
        orientation_list = [x,y,z,w]

        _,_,yaw = euler_from_quaternion(orientation_list)

        print(yaw)

if __name__ == "__main__":
    imu_sample = IMU()
    rospy.spin()
        