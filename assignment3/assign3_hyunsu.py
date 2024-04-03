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
        
        self.angle = 0
        self.warp_img_h = 460
        self.warp_img_w = 250
        self.warp_img_mid = 250 // 2
        self.prev_angle = []
        self.lane = np.array([0.0, 125.5])
        
        self.pub = rospy.Publisher("/lane_result", lane_info, queue_size=1)

    def warpping(self, image):
        source = np.float32([[270, 220], [360, 220], [110, 380], [510, 380]]) # 좌상, 우상, 좌하, 우하
        destination = np.float32([[0, 0], [250, 0], [0, 460], [250, 460]])

        transform_matrix = cv2.getPerspectiveTransform(source, destination)
        minv = cv2.getPerspectiveTransform(destination, source)
        _image = cv2.warpPerspective(image, transform_matrix, (250, 460))

        return _image, minv
    
    def to_canny(self, img, show=False):
        img = cv2.GaussianBlur(img, (7,7), 0)
        self.canny_low = 0
        self.canny_high = 100
        img = cv2.Canny(img, self.canny_low, self.canny_high)
        if show:
            cv2.imshow('canny', img)
        return img
    
    def hough(self, img, show=False):
        self.hough_threshold = 15
        self.min_gap = 10
        self.min_length = 20
        
        lines = cv2.HoughLinesP(img, 1, np.pi/180, self.hough_threshold, self.min_gap, self.min_length)
        if show:
            hough_img = np.zeros((img.shape[0], img.shape[1], 3))
            if lines is not None:
                for x1, y1, x2, y2 in lines[:, 0]:
                    cv2.line(hough_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imshow('hough', hough_img)
        return lines

    def filter(self, lines, show=False):
        '''
        filter lines that are close to previous angle and calculate its positions
        '''
        self.angle_tolerance = 30
        
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
                        cv2.line(filter_img, (x1, y1), (x2, y2), (255,0,0), 2)

        self.prev_angle.append(self.angle)
        if thetas:
            self.angle = np.mean(thetas)
        if show:
            cv2.imshow('filtered lines', filter_img)

        return positions
    
    def get_cluster(self, positions):
        '''
        group positions that are close to each other
        '''
        self.cluster_threshold = 10
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
            return
        
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
        self.lane = 0.5 * best + 0.5 * predicted_lane
        self.mid = np.mean(self.lane)
        
    
    def camera_callback(self, data):
        """
        code here
        """
        self.image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")        
        self.pub.publish(self.lane_detect())
    
    """
    lane detection preprocessing code here
    """
    def lane_detect(self):
        self.image = cv2.resize(self.image, (640, 480))
        cv2.imshow("original", self.image)
        
        # BEV warpped image
        warpped_img, minverse = self.warpping(self.image)
        cv2.imshow('warpped', warpped_img)
        
        # Canny Edge
        canny_img = self.to_canny(warpped_img, show=True)
        
        # Hough Transform
        lines = self.hough(canny_img, show=True)

        # Filtering
        positions = self.filter(lines, show=True)
        
        # Clustering
        lane_candidates = self.get_cluster(positions)
        
        
        # predict and update lane
        predicted_lane = self.predict_lane()
        self.update_lane(lane_candidates, predicted_lane)
        
        pub_msg = lane_info()
        pub_msg.left_x = int(self.lane[0])
        pub_msg.right_x = int(self.lane[-1])
        pub_msg.left_theta = self.angle
        pub_msg.right_theta = self.angle
        return pub_msg
        
        
    # def main(self):
    #     """
    #     code here(lane_detection)
    #     """
    #     pub_msg = lane_info()
    #     pub_msg.left_x = self.lane[0]
    #     pub_msg.right_x = self.lane[-1] 
    #     pub_msg.left_theta = self.angle
    #     pub_msg.right_theta = self.angle
    #     """
    #     code here(publish)
    #     """
    #     return pub_msg


if __name__ == "__main__":

    if not rospy.is_shutdown():
        lane_detect()
        rospy.spin()