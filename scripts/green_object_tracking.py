#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from sort.sort import Sort  # SORT 알고리즘을 이용한 트래커
import time

class ColorFilterTracker():
    def __init__(self):
        self.bridge = CvBridge()
        self.image = None
        self.index = 0
        self.previous_ids = set()
        self.last_index_time = 0
        
        self.lower = np.array([35, 75, 35])
        self.upper = np.array([75, 115, 75])

        self.tracker = Sort()
        
        rospy.init_node('color_filter_tracker_node', anonymous=False)
        rospy.Subscriber('/main_camera/image_raw/compressed', CompressedImage, self.camera_callback)

    def camera_callback(self, data):
        self.image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        self.main()

    def color_filter(self, image):
        # color filtering using RGB color space
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        white_mask = cv2.inRange(hls, self.lower, self.upper)
        masked = cv2.bitwise_and(image, image, mask=white_mask)
        return masked

    def get_bounding_boxes(self, mask):
        # bounding box detection using contour detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 80 and h > 60:  # minimum size of bounding box
                boxes.append([x, y, x + w, y + h])
        return boxes

    def main(self):
        if self.image is not None:
            resized_img = cv2.resize(self.image, (640, 480))
            filtered_img = self.color_filter(resized_img)
            
            # Detect bounding boxes
            gray_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY)
            boxes = self.get_bounding_boxes(thresh)

            current_time = time.time()

            # if bounding boxes are detected, update the tracker
            if len(boxes) > 0:
                tracked_objects = self.tracker.update(np.array(boxes))
            else:
                tracked_objects = []

            # Using SORT tracker, draw bounding boxes and object IDs
            current_ids = set()
            for obj in tracked_objects:
                x1, y1, x2, y2, obj_id = map(int, obj)
                current_ids.add(obj_id)
                
                # update the index if new object is detected
                if obj_id not in self.previous_ids and (current_time - self.last_index_time) >= 4:
                    print(f"New object detected with ID: {obj_id}")
                    self.index += 1
                    print("Mission Trigger Index:", self.index)
                    self.previous_ids.add(obj_id)
                    self.last_index_time = current_time
                
                cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.image, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("Tracked Image", self.image)
            cv2.waitKey(1)

if __name__ == "__main__":
    if not rospy.is_shutdown():
        ColorFilterTracker()
        rospy.spin()
