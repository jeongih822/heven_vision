#! /usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from collections import deque
from lane_control.msg import lane_info
from math import *
# lane_info
# int64 left_x
# int64 right_x
# float32 left_theta
# float32 right_theta


class lane_detect():
    def __init__(self):
        #변수들 모음
        # Warpping
        self.warp_x, self.warp_y = 240, 480
        self.warp_mid = self.warp_y/2
        # Canny_edge
        self.canny_low, self.canny_high = 100, 150
        # HoughlineP
        self.hough_threshold, self.min_length, self.min_gap = 10, 30, 10
        # Filtering
        self.center = self.warp_x/2
        self.angle_tolerance = np.radians(30)
        self.angle = 0.0
        self.left_angle = 0.0
        self.right_angle = 0.0
        self.prev_angle = deque([0.0], maxlen=5)
        # Clustering
        self.cluster_threshold = 25
        self.lane = np.array([20., 125., 230.])


        self.bridge = CvBridge()
        rospy.init_node('lane_detection_node', anonymous=False)
        rospy.Subscriber('/image_jpeg_2/compressed', CompressedImage, self.camera_callback)
        self.pub = rospy.Publisher("/lane_result", lane_info, queue_size=1)

    def camera_callback(self, data):
        self.img = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        self.main()
        self.pub.publish(self.lane_detect())
    
    def warpping(self, image):
        source = np.float32([[220, 280], [420, 280], [20, 480], [600, 480]])
        destination = np.float32([[0, 0], [self.warp_x, 0], [0, self.warp_y], [self.warp_x, self.warp_y]])
        transform_matrix = cv2.getPerspectiveTransform(source, destination)
        _image = cv2.warpPerspective(image, transform_matrix, (self.warp_x,self.warp_y))
        return _image

    def hough(self, img,show =False):
        lines = cv2.HoughLinesP(img, 1, np.pi/180, self.hough_threshold, self.min_gap, self.min_length)
        if show:
            hough_img = np.zeros((img.shape[0], img.shape[1], 3))
            if lines is not None:
                for x1, y1, x2, y2 in lines[:, 0]:
                    cv2.line(hough_img, (x1, y1), (x2, y2), red, 2)
            cv2.imshow('hough', hough_img) 
        return lines

    def filter(self, lines):
        thetas, positions = [], []
        left_thetas, right_thetas = [], []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                if y1 == y2:
                    continue
                line_center_x = (x1 + x2) / 2
                flag = 1 if y1-y2 > 0 else -1
                theta = np.arctan2(flag * (x2-x1), flag * 0.9* (y1-y2))
                if abs(theta - self.angle) < self.angle_tolerance:
                    position = float((x2-x1)*(self.warp_mid-y1))/(y2-y1) + x1
                    thetas.append(theta)
                    positions.append(position) 
                    if line_center_x < self.center: # Left lane
                        left_thetas.append(theta)
                    else:  # Right lane
                        right_thetas.append(theta)

        self.left_angle = np.mean(left_thetas) if left_thetas else None
        self.right_angle = np.mean(right_thetas) if right_thetas else None
        self.prev_angle.append(self.angle)
        if thetas:
            self.angle = np.mean(thetas)
        return positions

    def get_cluster(self, positions):
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
        predicted_lane = self.lane[1] + [-105/max(np.cos(self.angle), 0.75), 0, 105/max(np.cos(self.angle), 0.75)]
        predicted_lane = predicted_lane + (self.angle - np.mean(self.prev_angle))*70
        return predicted_lane

    def update_lane(self, lane_candidates, predicted_lane):
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
        self.lane = 0.7 * best + 0.3 * predicted_lane  #parameter
        self.left_x = self.lane[0]
        self.right_x = self.lane[2]


    def main(self):
        if self.img is not None: 
            self.img = cv2.resize(self.img, (640, 480))
            warpped_img = self.warpping(self.img)
            blurred_img = cv2.GaussianBlur(warpped_img, (0, 0), 1)
            canny_img = cv2.Canny(blurred_img, self.canny_low, self.canny_high)
            lines = self.hough(canny_img, show=False)
            positions = self.filter(lines)
            lane_candidates = self.get_cluster(positions)
            predicted_lane = self.predict_lane()
            self.update_lane(lane_candidates, predicted_lane)

                                           
        pub_msg = lane_info()
        pub_msg.left_x = np.int(self.left_x)
        pub_msg.right_x = np.int(self.right_x)
        pub_msg.left_theta = np.float32(self.left_angle)
        pub_msg.right_theta = np.float32(self.right_angle)
        return pub_msg
    
if __name__ == "__main__":
    if not rospy.is_shutdown():
        lane_detect()
        rospy.spin()