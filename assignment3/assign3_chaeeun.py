#! /usr/bin/env python3

import rospy
import cv2
import numpy as np
from collections import deque

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from lane_control.msg import lane_info
# lane_info
# int64 left_x
# int64 right_x
# float32 left_theta
# float32 right_theta
from math import *

red = (0, 0, 255)

class lane_detect():
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('lane_detection_node', anonymous=False)
        rospy.Subscriber('/image_jpeg_2/compressed', CompressedImage, self.camera_callback)
        self.pub = rospy.Publisher("/lane_result", lane_info, queue_size=1)

        self.warp_img_h, self.warp_img_w, self.warp_img_mid = 460, 250, 230

        # canny params
        self.canny_low, self.canny_high = 50, 80

        # HoughLineP params
        self.hough_threshold, self.min_length, self.min_gap = 10, 30, 10

        self.angle = 0.0
        self.prev_angle = deque([0.0], maxlen=5)

        self.lane = np.array([20.0, 125., 230.])

        # filtering params:
        self.angle_tolerance = np.radians(30)
        self.cluster_threshold = 25

        # custom message
        self.left_x = 0
        self.right_x = 0
        self.left_theta = 0.0
        self.right_theta = 0.0

    
    def camera_callback(self, data):
        self.image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        cv2.imshow("Original", self.image)

        # img = cv2.resize(self.image, (640, 480))

        self.pub.publish(self.main())
    
    """
    lane detection preprocessing code here
    """
    def warpping(self, image, show=False):
        """
            차선을 BEV로 변환하는 함수
            
            Return
            1) _image : BEV result image
            2) minv : inverse matrix of BEV conversion matrix
        """
        
        source = np.float32([[260, 255], [0, 330], [365, 255], [625, 330]])
        destination = np.float32([[128, 0], [128, 480], [512, 0], [512, 480]])

        transform_matrix = cv2.getPerspectiveTransform(source, destination)
        minv = cv2.getPerspectiveTransform(destination, source)
        _image = cv2.warpPerspective(image, transform_matrix, (640, 480))
        if show:
            cv2.imshow("warpped_img", _image)

        return _image, minv
    
    def to_canny(self, img, show=False):
        # img = cv2.GaussianBlur(img, (7,7), 0)
        img = cv2.Canny(img, self.canny_low, self.canny_high)
        if show:
            cv2.imshow('canny', img)
        return img

    def hough(self, img, show=False):
        lines = cv2.HoughLinesP(img, 1, np.pi/180, self.hough_threshold, self.min_gap, self.min_length)
        if show:
            hough_img = np.zeros((img.shape[0], img.shape[1], 3))
            if lines is not None:
                for x1, y1, x2, y2 in lines[:, 0]:
                    cv2.line(hough_img, (x1, y1), (x2, y2), red, 2)
            cv2.imshow('hough', hough_img)
        return lines

    def filter(self, lines, show=False):
        '''
        filter lines that are close to previous angle and calculate its positions
        '''
        thetas, positions = [], []
        if show:
            filter_img = np.zeros((self.warp_img_h, self.warp_img_w, 3))

        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                if y1 == y2:
                    continue
                flag = 1 if y1-y2 > 0 else -1
                theta = np.arctan2(flag * (x2-x1), flag * 0.9* (y1-y2))
                if abs(theta - self.angle) < self.angle_tolerance:
                    position = float((x2-x1)*(self.warp_img_mid-y1))/(y2-y1) + x1
                    thetas.append(theta)
                    positions.append(position) 
                    if show:
                        cv2.line(filter_img, (x1, y1), (x2, y2), red, 2)

        self.prev_angle.append(self.angle)
        if thetas:
            self.angle = np.mean(thetas)
        if show:
            cv2.imshow('filtered lines', filter_img)

        # compute left & right theta....??
        # if 0 <= position < 120:
        #     self.left_theta = self.angle
        # elif 120 <= position < 250:
        #     self.right_theta = self.angle
        return positions, self.angle

    def get_cluster(self, positions):
        '''
        group positions that are close to each other
        '''
        clusters = []
        for position in positions:
            if 0 <= position < 250:
                for cluster in clusters:
                    if abs(cluster[0] - position) < self.cluster_threshold:
                        cluster.append(position)
                        break
                else:
                    clusters.append([position])
        lane_candidates = [np.mean(cluster) for cluster in clusters]

        return lane_candidates

    def predict_lane(self):
        '''
        predicts lane positions from previous lane positions and angles
        '''
        predicted_lane = self.lane[1] + [-105/max(np.cos(self.angle), 0.75), 0, 105/max(np.cos(self.angle), 0.75)]
        predicted_lane = predicted_lane + (self.angle - np.mean(self.prev_angle))*70
        return predicted_lane

    def update_lane(self, lane_candidates, predicted_lane):
        '''
        calculate lane position using best fitted lane and predicted lane
        '''

        if not lane_candidates:
            self.lane = predicted_lane
            return self.lane

        possibles = []

        for lc in lane_candidates:

            idx = np.argmin(abs(self.lane-lc))

            if idx == 0:
                estimated_lane = [lc, lc + 105/max(np.cos(self.angle), 0.75), lc + 210/max(np.cos(self.angle), 0.75)]
                lc2_candidate, lc3_candidate = [], []
                for lc2 in lane_candidates:
                    if abs(lc2-estimated_lane[1]) < 50 :
                        lc2_candidate.append(lc2)
                for lc3 in lane_candidates:
                    if abs(lc3-estimated_lane[2]) < 50 :
                        lc3_candidate.append(lc3)
                if not lc2_candidate:
                    lc2_candidate.append(estimated_lane[1])
                if not lc3_candidate:
                    lc3_candidate.append(estimated_lane[2])
                for lc2 in lc2_candidate:
                    for lc3 in lc3_candidate:
                        possibles.append([lc, lc2, lc3])

            elif idx == 1:
                estimated_lane = [lc - 105/max(np.cos(self.angle), 0.75), lc, lc + 105/max(np.cos(self.angle), 0.75)]
                lc1_candidate, lc3_candidate = [], []
                for lc1 in lane_candidates:
                    if abs(lc1-estimated_lane[0]) < 50 :
                        lc1_candidate.append(lc1)
                for lc3 in lane_candidates:
                    if abs(lc3-estimated_lane[2]) < 50 :
                        lc3_candidate.append(lc3)
                if not lc1_candidate:
                    lc1_candidate.append(estimated_lane[0])
                if not lc3_candidate:
                    lc3_candidate.append(estimated_lane[2])
                for lc1 in lc1_candidate:
                    for lc3 in lc3_candidate:
                        possibles.append([lc1, lc, lc3])

            else :
                estimated_lane = [lc - 210/max(np.cos(self.angle), 0.75), lc - 105/max(np.cos(self.angle), 0.75), lc]
                lc1_candidate, lc2_candidate = [], []
                for lc1 in lane_candidates:
                    if abs(lc1-estimated_lane[0]) < 50 :
                        lc1_candidate.append(lc1)
                for lc2 in lane_candidates:
                    if abs(lc2-estimated_lane[1]) < 50 :
                        lc2_candidate.append(lc2)
                if not lc1_candidate:
                    lc1_candidate.append(estimated_lane[0])
                if not lc2_candidate:
                    lc2_candidate.append(estimated_lane[1])
                for lc1 in lc1_candidate:
                    for lc2 in lc2_candidate:
                        possibles.append([lc1, lc2, lc])

        possibles = np.array(possibles)
        error = np.sum((possibles-predicted_lane)**2, axis=1)
        best = possibles[np.argmin(error)]
        self.lane = 0.7 * best + 0.3 * predicted_lane
        self.mid = np.mean(self.lane)
        
        return self.lane
        


    # def mark_lane(self, img, lane=None):
    #     '''
    #     mark calculated lane position to an image 
    #     '''
    #     # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #     if lane is None:
    #         lane = self.lane
    #         self.mid = self.lane[1]
    #     l1, l2, l3 = self.lane
    #     cv2.circle(img, (int(l1), 230), 3, red, 5, cv2.FILLED)
    #     cv2.circle(img, (int(l2), 230), 3, green, 5, cv2.FILLED)
    #     cv2.circle(img, (int(l3), 230), 3, blue, 5, cv2.FILLED)
    #     cv2.imshow('marked', img)

    
    def main(self):
        """
        code here(lane_detection)
        """
        # self.image = cv2.resize(self.image, (640, 480))
        # cv2.imshow("original", self.image)
        
        # BEV warpped image
        warpped_img, minverse = self.warpping(self.image, show=True)
        # cv2.imshow('warpped', warpped_img)
        
        # Canny Edge
        canny_img = self.to_canny(wargpped_im, show=True)
        
        # Hough Transform
        lines = self.hough(canny_img, show=True)

        # Filtering
        positions, angle = self.filter(lines, show=True)
        
        # Clustering
        lane_candidates = self.get_cluster(positions)
        
        # predict and update lane
        predicted_lane = self.predict_lane()
        lane = self.update_lane(lane_candidates, predicted_lane)
        
        left_x = lane[0]
        right_x = lane[2]
        left_angle = angle
        right_angle = angle
        
        cv2.waitKey(1)
        
        pub_msg = lane_info()
        pub_msg.left_x = left_x
        pub_msg.right_x = right_x
        pub_msg.left_theta = left_angle
        pub_msg.right_theta = right_angle

        return pub_msg


if __name__ == "__main__":

    if not rospy.is_shutdown():
        lane_detect()
        rospy.spin()