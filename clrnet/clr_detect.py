import numpy as np
import torch
import cv2
import os
import os.path as osp
import glob
import argparse
from clrnet.datasets.process import Process
from clrnet.models.registry import build_net
from clrnet.utils.config import Config
from clrnet.utils.visualization import imshow_lanes
from clrnet.utils.net_utils import load_network
from pathlib import Path
from tqdm import tqdm

class Detect(object):
    def __init__(self):
        self.cfg = Config.fromfile('./configs/clrnet/clr_resnet101_tusimple.py')
        self.processes = Process(self.cfg.val_process, self.cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids = range(1)).cuda()
        self.net.eval()
        load_network(self.net, './clr_final.pth')

    def preprocess(self, ori_img):
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'ori_img':ori_img})
        return data

    def inference(self, data):
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.heads.get_lanes(data)
        return data

    def show(self, data):
        # out_file = self.cfg.savedir 
        # if out_file:
        #     out_file = osp.join(out_file, osp.basename(data['img_path']))
        # print(type(data))
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        imshow_lanes(data['ori_img'], lanes)

    def run(self, data):
        data = self.preprocess(data)
        data['lanes'] = self.inference(data)[0]
        # if self.cfg.show or self.cfg.savedir:
        self.show(data)
        return data

def process():
    detect = Detect()
    cap = cv2.VideoCapture('./2_test_video.mp4')
    # cap = cv2.VideoCapture('./3_test_video.mp4')
    while True:
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
        print(frame.shape)
        detect.run(frame)

if __name__ == '__main__':
    process()