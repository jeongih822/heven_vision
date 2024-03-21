#! /usr/bin/env python3

import rospy
import cv2

from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

class example():
    def __init__(self):
        
        self.bridge = CvBridge()
        
        rospy.init_node('CvBride_node', anonymous=False)
        rospy.Subscriber('/image_jpeg_2/compressed', CompressedImage, self.camera_callback)
        
    def color_filter(self, image):

        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        
        lower = np.array([15, 90, 15])
        upper = np.array([255, 255, 255])

        white_mask = cv2.inRange(hls, lower, upper)
        masked = cv2.bitwise_and(image, image, mask = white_mask)

        return masked

    def warpping(self, image):
        source = np.float32([[235, 250], [365, 222], [170, 340], [500, 350]])
        destination = np.float32([[0, 0], [250, 0], [0, 460], [250, 460]])

        transform_matrix = cv2.getPerspectiveTransform(source, destination)
        minv = cv2.getPerspectiveTransform(destination, source)
        _image = cv2.warpPerspective(image, transform_matrix, (250, 460))

        return _image, minv

    def plothistogram(self, image):
        histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
        midpoint = np.int_(histogram.shape[0]/2)
        leftbase = np.argmax(histogram[:midpoint])
        rightbase = np.argmax(histogram[midpoint:]) + midpoint

        return leftbase, rightbase, histogram

    def slide_window_search(self, binary_warped, left_current, right_current):
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))

        nwindows = 24
        window_height = np.int_(binary_warped.shape[0] / nwindows)
        nonzero = binary_warped.nonzero()  # 선이 있는 부분의 인덱스만 저장
        nonzero_y = np.array(nonzero[0])  # 선이 있는 부분 y의 인덱스 값
        nonzero_x = np.array(nonzero[1])  # 선이 있는 부분 x의 인덱스 값
        margin = 100
        minpix = 50
        left_lane = []
        right_lane = []
        color = [0, 255, 0]
        thickness = 2

        for w in range(nwindows):
            win_y_low = binary_warped.shape[0] - (w + 1) * window_height  # window 윗부분
            win_y_high = binary_warped.shape[0] - w * window_height  # window 아랫 부분
            win_xleft_low = left_current - margin  # 왼쪽 window 왼쪽 위
            win_xleft_high = left_current + margin  # 왼쪽 window 오른쪽 아래
            win_xright_low = right_current - margin  # 오른쪽 window 왼쪽 위
            win_xright_high = right_current + margin  # 오른쪽 window 오른쪽 아래

            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
            good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
            left_lane.append(good_left)
            right_lane.append(good_right)
            cv2.namedWindow('sliding window')
            cv2.moveWindow('sliding window', 650, 0)
            cv2.imshow("sliding window", out_img)

            if len(good_left) > minpix:
                left_current = np.int_(np.mean(nonzero_x[good_left]))
            if len(good_right) > minpix:
                right_current = np.int_(np.mean(nonzero_x[good_right]))

        left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침
        right_lane = np.concatenate(right_lane)

        leftx = nonzero_x[left_lane]
        lefty = nonzero_y[left_lane]
        rightx = nonzero_x[right_lane]
        righty = nonzero_y[right_lane]


        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        
        ltx = np.trunc(left_fitx)  # np.trunc() -> 소수점 부분을 버림
        rtx = np.trunc(right_fitx)
        
        out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
        out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]

        ret = {'left_fitx' : ltx, 'right_fitx': rtx, 'ploty': ploty}

        return ret

    def draw_lane_lines(self, original_image, warped_image, Minv, draw_info):
        left_fitx = draw_info['left_fitx']
        right_fitx = draw_info['right_fitx']
        ploty = draw_info['ploty']

        warp_zero = np.zeros_like(warped_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        pts_left_list = pts_left.tolist()
        pts_right_list = pts_right.tolist()

        mean_x = np.mean((left_fitx, right_fitx), axis=0)
        pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))
        cv2.fillPoly(color_warp, np.int_([pts_mean]), (0, 0, 255))

        for left_point in pts_left_list[0]:
            cv2.line(color_warp, (round(left_point[0]), round(left_point[1])), (round(left_point[0]), round(left_point[1])), (0, 255, 0), 5)
            
        for right_point in pts_right_list[0]:
            cv2.line(color_warp, (round(right_point[0]), round(right_point[1])), (round(right_point[0]), round(right_point[1])), (0, 255, 0), 5)
        
        newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))        
            
        result = cv2.addWeighted(original_image, 1, newwarp, 0.4, 0)

        return result, newwarp, color_warp, pts_left_list, pts_right_list
    def camera_callback(self, data):
        self.image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        self.image = cv2.resize(self.image, (640, 480))
            
        # BEV wrapped img
        warpped_img, minverse = self.warpping(self.image)
        # cv2.namedWindow('warpped')
        # cv2.moveWindow('warpped', 0, 0)
        # cv2.imshow('warpped', warpped_img)

        blurred_img = cv2.GaussianBlur(warpped_img, (0, 0), 1)
        # cv2.namedWindow('Gaussian Blur')
        # cv2.moveWindow('Gaussian Blur', 350, 0)
        # cv2.imshow('Gaussian Blur', blurred_img)
        
        # BEV 필터링
        w_f_img = self.color_filter(warpped_img)
        # cv2.namedWindow('Color filter')
        # cv2.moveWindow('Color filter', 0, 550)
        # cv2.imshow('Color filter', w_f_img)

        # BEV  threshold
        _gray = cv2.cvtColor(w_f_img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(_gray, 170, 255, cv2.THRESH_BINARY)
        cv2.namedWindow('threshold')
        cv2.moveWindow('threshold', 350, 550)
        cv2.imshow('threshold', thresh)
        
        # 선 분포도 조사 histogram
        leftbase, rightbase, hist = self.plothistogram(thresh)
        # plt.plot(hist)
        # plt.show()

        # histogram 기반 Sliding Window Search
        draw_info = self.slide_window_search(thresh, leftbase, rightbase)

        # 원본 이미지에 라인 넣기
        result, newwarp, color_warp, pts_left_list, pts_right_list = self.draw_lane_lines(self.image, thresh, minverse, draw_info)
        # cv2.namedWindow('newwarp')
        # cv2.moveWindow('newwarp', 1200, 550)
        # cv2.imshow("newwarp", newwarp)
        
        # cv2.namedWindow('colorwarp')
        # cv2.moveWindow('colorwarp', 1200, 0)
        # cv2.imshow("colorwarp", color_warp)
        
        cv2.namedWindow('result')
        cv2.moveWindow('result', 650, 550)
        cv2.imshow("result", result)
        # cv2.imshow("Display", self.image)
        cv2.waitKey(10)

if __name__ == "__main__":
    
    if not rospy.is_shutdown():
        example()
        rospy.spin()