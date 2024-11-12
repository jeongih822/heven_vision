import cv2
import numpy as np
from math import *
import matplotlib.pyplot as plt

angle = []

def color_filter(image):
    """
        HLS 필터 사용
        
        lower & upper : 흰색으로 판단할 minimum pixel 값
        white_mask : lower과 upper 사이의 값만 남긴 mask
        masked : cv2.bitwise_and() 함수를 통해 흰색인 부분 제외하고는 전부 검정색 처리
    """
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    
    lower = np.array([40, 185, 10])
    upper = np.array([255, 255, 255])

    white_mask = cv2.inRange(hls, lower, upper)
    masked = cv2.bitwise_and(image, image, mask = white_mask)

    return masked

class lane_detect():
    def __init__(self):
        self.image_orig = None
        self.cluster_threshold = 30
        path = '../video/lane_test.mp4'
        self.cap = cv2.VideoCapture(path)  # Capture from the default camera

        if not self.cap.isOpened():
            print("Error: Could not open video capture.")
            exit()

    def warpping(self, image):
        """
        차선을 BEV로 변환하는 함수
        
        Return:
            1) _image : BEV result image
            2) minv : inverse matrix of BEV conversion matrix
        """
        source = np.float32([[235, 250], [330, 250], [80, 475], [460, 475]])
        destination = np.float32([[0, 0], [250, 0], [0, 460], [250, 460]])

        transform_matrix = cv2.getPerspectiveTransform(source, destination)
        minv = cv2.getPerspectiveTransform(destination, source)
        _image = cv2.warpPerspective(image, transform_matrix, (250, 460))

        return _image, minv

    def get_cluster(self, positions):
        '''
        group positions that are close to each other
        '''
        clusters = []
        for position in positions:
            if 5 <= position < 315:
                for cluster in clusters:
                    if abs(cluster[0] - position) < self.cluster_threshold:
                        cluster.append(position)
                        break
                else:
                    clusters.append([position])
        lane_candidates = [np.mean(cluster) for cluster in clusters]

        return lane_candidates

    def high_level_detect(self, hough_img):
        nwindows = 10  # window 개수
        margin = 50
        minpix = 30  # 차선 인식을 판정하는 최소 픽셀 수

        # 아래 30%만 histogram 계산
        histogram = np.sum(hough_img[hough_img.shape[0] // 10 * 8:], axis=0)

        midpoint = np.int32(histogram.shape[0] / 2)

        # 왼쪽 절반에서 0이 아닌 값만 선택
        left_non_zero_indices = np.where(histogram[:midpoint] != 0)[0]
        if len(left_non_zero_indices) == 0:
            leftx_current = 160
        else:
            left_top_five_indices = np.argpartition(histogram[:midpoint], -5)[-5:]
            left_top_five_indices = left_top_five_indices[np.argsort(histogram[:midpoint][left_top_five_indices])][::-1]
            leftx_current_list = np.array([left_top_five_indices[i] for i in range(5)])
            leftx_current = np.max(leftx_current_list)

        # 오른쪽 절반에서 0이 아닌 값만 선택
        right_non_zero_indices = np.where(histogram[midpoint:] != 0)[0]
        if len(right_non_zero_indices) == 0:
            rightx_current = 480
        else:
            right_top_five_indices = np.argpartition(histogram[midpoint:], -5)[-5:]
            right_top_five_indices = right_top_five_indices[np.argsort(histogram[midpoint:][right_top_five_indices])][::-1]
            rightx_current_list = np.array([right_top_five_indices[i] for i in range(5)])
            rightx_current = np.min(rightx_current_list) + midpoint

        save_leftx = leftx_current
        save_rightx = rightx_current
        window_height = np.int32(hough_img.shape[0] / nwindows)
        
        nz = hough_img.nonzero()

        left_lane_inds = []
        right_lane_inds = []
        
        global lx, ly, rx, ry
        lx, ly, rx, ry = [], [], [], []

        global out_img
        out_img = np.dstack((hough_img, hough_img, hough_img)) * 255

        left_sum = 0
        right_sum = 0

        total_loop = 0

        for window in range(nwindows - 4):
            win_yl = hough_img.shape[0] - (window + 1) * window_height
            win_yh = hough_img.shape[0] - window * window_height

            win_xll = leftx_current - margin
            win_xlh = leftx_current + margin
            win_xrl = rightx_current - margin
            win_xrh = rightx_current + margin

            cv2.rectangle(out_img, (win_xll, win_yl), (win_xlh, win_yh), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xrl, win_yl), (win_xrh, win_yh), (0, 255, 0), 2)

            good_left_inds = ((nz[0] >= win_yl) & (nz[0] < win_yh) & (nz[1] >= win_xll) & (nz[1] < win_xlh)).nonzero()[0]
            good_right_inds = ((nz[0] >= win_yl) & (nz[0] < win_yh) & (nz[1] >= win_xrl) & (nz[1] < win_xrh)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            cluster_left = self.get_cluster(nz[1][good_left_inds])
            if len(good_left_inds) > minpix and len(cluster_left) > 0:
                leftx_current = np.int32(np.max(cluster_left))
            elif len(good_left_inds) > minpix and len(cluster_left) == 0:
                leftx_current = np.int32(np.mean(nz[1][good_left_inds]))

            lx.append(leftx_current)
            ly.append((win_yl + win_yh) / 2)

            left_sum += leftx_current

            cluster_right = self.get_cluster(nz[1][good_right_inds])
            if len(good_right_inds) > minpix and len(cluster_right) > 0:
                rightx_current = np.int32(np.min(cluster_right))
            elif len(good_right_inds) > minpix and len(cluster_right) == 0:
                rightx_current = np.int32(np.mean(nz[1][good_right_inds]))

            rx.append(rightx_current)
            ry.append((win_yl + win_yh) / 2)

            right_sum += rightx_current

            total_loop += 1

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        lfit = np.polyfit(np.array(ly[1:]), np.array(lx[1:]), 2)
        rfit = np.polyfit(np.array(ry[1:]), np.array(rx[1:]), 2)

        out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
        out_img[nz[0][right_lane_inds], nz[1][right_lane_inds]] = [0, 0, 255]

        left_avg = left_sum / total_loop
        right_avg = right_sum / total_loop

        if save_leftx == 160:
            left_avg = right_avg - 320

        if save_rightx == 480:
            right_avg = left_avg + 320

        return lfit, rfit, left_avg, right_avg
    
    def lane_detect(self):
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Error: Failed to capture frame.")
                return

            self.image = cv2.resize(frame, (640, 480))

            if self.image_orig is not None:
                self.image_orig = cv2.resize(self.image_orig, (640, 480))
                cv2.polylines(self.image_orig, [np.array(self.source)], True, (255, 0, 255), 2)
                warpped_img_orig, minv_orig = self.warpping(self.image_orig)

            warpped_img, minv = self.warpping(self.image)

            blurred_img = cv2.GaussianBlur(warpped_img, (0, 0), 1)

            w_f_img = color_filter(warpped_img)

            _gray = cv2.cvtColor(w_f_img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(_gray, 170, 255, cv2.THRESH_BINARY)

            # left_fit, right_fit, ignore_right, l_avg, r_avg = self.high_level_detect(hough_img)
            left_fit, right_fit, l_avg, r_avg = self.high_level_detect(thresh)
            
            left_fit = np.polyfit(np.array(ly),np.array(lx),1)
            right_fit = np.polyfit(np.array(ry),np.array(rx),1)
            # print("left_fit: ", left_fit)
            # print("right_fit: ", right_fit)
            
            line_left = np.poly1d(left_fit)
            line_right = np.poly1d(right_fit)
            # print("line_left: ", line_left)
            # print("line_right: ", line_right)


            # 좌,우측 차선의 휘어진 각도
            left_line_angle = degrees(atan(line_left[1])) + 90
            right_line_angle = degrees(atan(line_right[1])) + 90
            
            shift_const = 320
    
            final_left_angle = left_line_angle  
            final_right_angle = right_line_angle
            # cv2.namedWindow('Sliding Window')
            # cv2.moveWindow('Sliding Window', 1400, 0)

            print("left avg : %3f   right avg : %3f" %(l_avg, r_avg))
            print("left_th : %3f   right_th : %3f" %(left_line_angle, right_line_angle))
            
            cv2.imshow("Sliding Window", out_img)
            cv2.waitKey(1)

            if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    LaneDetect = lane_detect()
    LaneDetect.lane_detect()