#! /usr/bin/env python3

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from lane_control.msg import lane_info
from math import *

def region_of_interest(img, vertices, color3=(255,255,255), color1=255):

    mask = np.zeros_like(img) 
    
    if len(img.shape) > 2:
        color = color3
    else: 
        color = color1

    vertices = np.int32(vertices)
    cv2.fillPoly(mask, vertices, color)
    
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def warpping(image):
    """
        차선을 BEV로 변환하는 함수
        
        Return
        1) _image : BEV result image
        2) minv : inverse matrix of BEV conversion matrix
    """

    # roi_source = np.float32([[86, 150], [554, 150], [640, 400], [0, 400]])
    roi_source = np.float32([[120, 0], [520, 0], [520, 480], [120, 480]])
    # source = np.float32([[200, 210], [20,480], [420,210], [620, 480]])
    source = np.float32([[260, 255], [0, 330], [365, 255], [625, 330]])
    destination = np.float32([[128, 0], [128, 480], [512, 0], [512, 480]])
    
    M = cv2.getPerspectiveTransform(source, destination)
    Minv = cv2.getPerspectiveTransform(destination, source)
    
    # image = region_of_interest(image, [roi_source])
    
    warp_image = cv2.warpPerspective(image, M, (640, 480), flags=cv2.INTER_LINEAR)
    warp_image = region_of_interest(warp_image, [roi_source])
    # cv2.rectangle(warp_image, (195, 445), (480, 480), (0, 0, 0), -1)

    return warp_image, Minv

def color_filter(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    lower = np.array([150, 120, 115])
    upper = np.array([255, 255, 255])
    
    # lower = np.array([40, 185, 10])
    # upper = np.array([255, 255, 255])

    # yellow_lower = np.array([0, 85, 81])
    # yellow_upper = np.array([190, 255, 255])

    # yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(image, lower, upper)
    # mask = cv2.bitwise_or(yellow_mask, white_mask)
    masked = cv2.bitwise_and(image, image, mask = white_mask)
    
    return masked


class lane_detect():
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('lane_detection_node', anonymous=False)
        rospy.Subscriber('/image_jpeg_2/compressed', CompressedImage, self.camera_callback)
        self.pub = rospy.Publisher("/lane_result", lane_info, queue_size=1)

    
    def camera_callback(self, data):
        self.image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        cv2.namedWindow('Original')
        cv2.moveWindow('Original', 0, 600)
        cv2.imshow('Original', self.image)
        self.pub.publish(self.lane_detect())
        # self.lane_detect()

    
    def high_level_detect(self, hough_img):

        nwindows = 10       # window 개수
        margin = 75         # window 가로 길이
        minpix = 30          # 차선 인식을 판정하는 최소 픽셀 수
        lane_bin_th = 225
       
        histogram = np.sum(hough_img[hough_img.shape[0]//2:,:],   axis=0)
        
        midpoint = np.int32(histogram.shape[0]/2)
    
        leftx_current = 160
        rightx_current = 480
        # leftx_current = np.argmax(histogram[:midpoint])
        # rightx_current = np.argmax(histogram[midpoint:]) + midpoint

        # print("left_cur : %.3f, right_cur : %.3f" % (leftx_current, rightx_current))

        # 차량 주행 중에 차선인식이 일어나지 않는 경우,
        # 해당 차선을 인식할 때 사용되던 window들의 default 위치를 조정.
        if leftx_current == 0:
            leftx_current = int(midpoint - 355 / 2)
                        
        if rightx_current == midpoint:
            rightx_current = int(midpoint + 355 / 2)

        # 쌓을 window의 height 설정
        window_height = np.int32(hough_img.shape[0]/nwindows)
        
        # 240*320 픽셀에 담긴 값중 0이 아닌 값을 저장한다.
        # nz[0]에는 index[row][col] 중에 row파트만 담겨있고 nz[1]에는 col이 담겨있다.
        nz = hough_img.nonzero()

        left_lane_inds = []
        right_lane_inds = []

        global lx, ly, rx, ry
        lx, ly, rx, ry = [], [], [], []

        global out_img
        out_img = np.dstack((hough_img, hough_img, hough_img))*255

        ignore_right = False
        cnt = 0
        
        left_sum = 0
        right_sum = 0

        total_loop = 0

        for window in range(nwindows-4):
            
            # bounding box 크기 설정
            win_yl = hough_img.shape[0] - (window+1)*window_height
            win_yh = hough_img.shape[0] - window*window_height

            win_xll = leftx_current - margin
            win_xlh = leftx_current + margin
            win_xrl = rightx_current - margin
            win_xrh = rightx_current + margin

            # out image에 bounding box 시각화
            cv2.rectangle(out_img,(win_xll,win_yl),(win_xlh,    win_yh),    (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xrl,win_yl),(win_xrh,    win_yh),    (0,255,0), 2) 

            # 흰점의 픽셀들 중에 window안에 들어오는 픽셀인지 여부를 판단하여 
            # good_left_inds와 good_right_inds에 담는다.
            good_left_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&   (nz    [1] >= win_xll)&(nz[1] < win_xlh)).nonzero()    [0]
            good_right_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)   &(nz   [1] >= win_xrl)&(nz[1] < win_xrh)).nonzero()    [0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # nz[1]값들 중에 good_left_inds를 index로 삼는 nz[1]들의 평균을 구해서 leftx_current를 갱신한다.
            if len(good_left_inds) > minpix:
                leftx_current = np.int32(np.mean(nz[1]    [good_left_inds])   )

            #lx ly rx ry에 x,y좌표들의 중심점들을 담아둔다.
            lx.append(leftx_current)
            ly.append((win_yl + win_yh)/2)

            left_sum += leftx_current

            # nz[1]값들 중에 good_right_inds를 index로 삼는 nz[1]들의 평균을 구해서 rightx_current를 갱신한다.            
            if len(good_right_inds) > minpix:        
                rightx_current = np.int32(np.mean(nz[1]       [good_right_inds]))

            rx.append(rightx_current)
            ry.append((win_yl + win_yh)/2)

            right_sum += rightx_current
            
            # if cnt < 2:    
            #     # nz[1]값들 중에 good_right_inds를 index로 삼는 nz[1]들의 평균을 구해서 rightx_current를 갱신한다.            
            #     if len(good_right_inds) > minpix:        
            #         rightx_current = np.int32(np.mean(nz[1]       [good_right_inds]))
            #         rx.append(rightx_current)
            #         ry.append((win_yl + win_yh)/2)
                    
                
            #     else:
            #         if window < 2:
            #             cnt += 1
                    
            #         rx.append(int(midpoint + 355 / 2))
            #         ry.append((win_yl + win_yh)/2)
            
            # else:
            #     rx.append(int(midpoint + 355 / 2))
            #     ry.append((win_yl + win_yh)/2)
                
            #     ignore_right = True
            
            # print("현재 WINDOW : ", window)
            # print("right x cur : ", rightx_current)
            # print("cnt : ", cnt)
            # print("======================")
            total_loop += 1

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        lfit = np.polyfit(np.array(ly[1:]),np.array(lx[1:]),2)
        rfit = np.polyfit(np.array(ry[1:]),np.array(rx[1:]),2)

        #out_img에서 왼쪽 선들의 픽셀값을 BLUE로, 
        #오른쪽 선들의 픽셀값을 RED로 바꿔준다.
        out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
        out_img[nz[0][right_lane_inds] , nz[1][right_lane_inds]]= [0, 0, 255]    

        left_avg = left_sum / total_loop
        right_avg = right_sum / total_loop

        return lfit, rfit, ignore_right, left_avg, right_avg
    
    def lane_detect(self):
        
        self.image = cv2.resize(self.image, (640, 480))
        
        # cv2.namedWindow('Original')
        # cv2.moveWindow('Original', 700, 0)
        # cv2.imshow('Original', self.image)
        
        warpped_img, minv = warpping(self.image)
        cv2.namedWindow('BEV')
        cv2.moveWindow('BEV', 0, 0)
        cv2.imshow('BEV', warpped_img)
        
        blurred_img = cv2.GaussianBlur(warpped_img, (7, 7), 5)
        cv2.namedWindow('Blurred')
        cv2.moveWindow('Blurred', 350, 0)
        cv2.imshow('Blurred', blurred_img)
        
        w_f_img = color_filter(blurred_img)
        cv2.namedWindow('Color filter')
        cv2.moveWindow('Color filter', 0, 550)
        cv2.imshow('Color filter', w_f_img)
        
        grayscale = cv2.cvtColor(w_f_img, cv2.COLOR_BGR2GRAY)
        
        ret, thresh = cv2.threshold(grayscale, 170, 255, cv2.THRESH_BINARY) #170, 255
        
        canny_img = cv2.Canny(thresh, 10, 100)
        cv2.namedWindow('Canny')
        cv2.moveWindow('Canny', 700, 600)
        cv2.imshow('Canny', canny_img)
        
        lines = cv2.HoughLines(canny_img, 1, np.pi/180, 80, None, 0, 0)
        
        # hough_img = thresh.copy()
        hough_img = canny_img.copy()
        # hough_img = np.zeros((640, 480))
        
        if lines is not None:
            for line in lines:
                
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int (x0 + 1000*(-b))
                y1 = int ((y0) + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                
                slope = 90 - degrees(atan(b / a))
            
                if abs(slope) < 20:
                    cv2.line(hough_img, (x1, y1), (x2, y2), 0, 30)
                
                else:    
                    cv2.line(hough_img, (x1, y1), (x2, y2), 255, 8)
        
        cv2.namedWindow('Hough')
        cv2.moveWindow('Hough', 700, 0)
        cv2.imshow('Hough', hough_img)
        
        left_fit, right_fit, ignore_right, l_avg, r_avg = self.high_level_detect(hough_img)
        
        left_fit = np.polyfit(np.array(ly),np.array(lx),1)
        right_fit = np.polyfit(np.array(ry),np.array(rx),1)
        
        line_left = np.poly1d(left_fit)
        line_right = np.poly1d(right_fit)
        
        # 좌,우측 차선의 휘어진 각도
        left_line_angle = degrees(atan(line_left[1])) + 90
        right_line_angle = degrees(atan(line_right[1])) + 90

        #print("left : %3f   right : %3f" %(left_fit, right_fit))

        # print("left lane : ", line_left)
        # print("right lane : ", line_right)

        # print("left : %3f   right : %3f" %(left_line_angle, right_line_angle))
        
        shift_const = 320
   
        # 좌,우측 차선에서 떨어진 거리
        left_position = ((shift_const - float(lx[2])) / shift_const) * 2 - 1
        right_position = -((-shift_const + float(rx[2])) / shift_const) * 2 + 1

        # print("left_pos : %3f   right_pos : %3f" %(left_position, right_position))
          
        # 현재 위치 변수
        position = 0
        
        # 우측 차선이 인식되지 않으면 좌측 차선을 위주로 현재 위치 계산
        if abs(degrees(atan(line_right[1]))) < 0.05:
            if ignore_right == True:
                position = left_position
            else:
                position = 0.9*left_position + 0.1*right_position
        
        # 좌측 차선이 인식되지 않으면 우측 차선을 위주로 현재 위치 계산
        elif abs(degrees(atan(line_left[1]))) < 0.05:
            position = 0.1*left_position + 0.9*right_position
        
        # 두 차선이 모두 인식되면 두 차선에서 떨어진 거리로 현재 위치 계산
        else:
            position = (left_position + right_position) / 2
        
        #print("left_pos : %.3f   right_pos : %.3f" %(left_position, right_position))
        # print("cur_pos : %.3f" % (position))

        cv2.namedWindow('Sliding Window')
        cv2.moveWindow('Sliding Window', 1400, 0)
        cv2.imshow("Sliding Window", out_img)
        cv2.waitKey(1)

        pub_msg = lane_info()
        pub_msg.left_x = int(abs(shift_const - l_avg))
        pub_msg.right_x = int(abs(r_avg - shift_const))
        pub_msg.left_theta = left_line_angle
        pub_msg.right_theta = right_line_angle

        # print("left avg : %3f   right avg : %3f" %(l_avg, r_avg))
        print("left_th : %3f   right_th : %3f" %(left_line_angle, right_line_angle))
        print("left : %3f   right : %3f    sum : %3f" %(pub_msg.left_x, pub_msg.right_x, pub_msg.left_x + pub_msg.right_x))

        return pub_msg

if __name__ == "__main__":

    if not rospy.is_shutdown():
        lane_detect()
        rospy.spin()