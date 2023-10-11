#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@file solo_detect.py
@brief 即時顯示當前相機影像&SOLOv2預測結果，
       將SOLOv2遮罩publish到/solo_mask topic上

參數設定：
(1) show_result = True  #即時顯示RGB影像及SOLOv2遮罩
(2) show_mask = True    #只顯示SOLOv2遮罩
(3) save_result = False #儲存RGB影像及SOLOv2遮罩
    影像儲存位置：./src/solo_detect/test_img/solo_res_{}.jpg
(4) use_test_img = False#使用test_img
(5) config_file         #SOLOv2訓練設定檔
(6) checkpoint_file     #SOLOv2權重檔
'''

import cv2
import sys
sys.path.insert(1, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
Workspace_Path = sys.path[0].rsplit('/',2)[0]
sys.path.append(Workspace_Path)
sys.path.append(Workspace_Path+'/mmdetection2/')
from cv_bridge import CvBridge, CvBridgeError

from mmdet.apis import init_detector, inference_detector
import mmcv

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int32MultiArray

import numpy as np
import os

import time

#=====Parameters Setting=====#
show_result = True
show_mask = True
save_result = False

ws_path = Workspace_Path #'/home/robotarm/forklift_pallet_ws/src'
test_img_path = ws_path + '/solo_detect/test_img/altek_img_1.jpg'

checkpoint_file = ws_path + '/mmdetection2/work_dirs/pallet.pth' #SOLO/configs/solov2/***.py
config_file = ws_path + '/mmdetection2/configs/solov2/pallet_test_2.py'                      #SOLO/data/***.pth
#=====Parameters Setting=====#

cnt = 0

class SOLO_Det:
    def __init__(self) -> None:
        rospy.init_node("SOLO_Node")
        # build the model from a config file and a checkpoint file
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')
        
        self.bridge = CvBridge()
        rospy.Subscriber("/camera/color/image_raw", Image, self.imageCallback)
        rospy.spin()

    def imageCallback(self, img_msg):
        global cnt

        # cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        # cv_image = cv2.imread("/home/iclab/work/src/pallet_seg/src/test_img.jpg")
        cv_image = cv2.imread("/home/iclab/work/src/pallet_seg/test_img/altek_img_1.jpg")
        h, w, _ = cv_image.shape

        result = inference_detector(self.model, cv_image)

        num_mask = len(result)
        np.random.seed(42)
        color_masks = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(num_mask)
        ]
        # print(len(result), len(result[0][0]), len(result[1][0]), result[0][0], result[1][0])
        # print()
        cur_mask = result[1][0][0]
        print(cur_mask.shape)
        # cur_mask = cv2.imresize(cur_mask, (w, h))
        # print(cur_mask)      
        cur_mask = (cur_mask>0.5).astype(np.uint8)
        cur_mask_bool = cur_mask.astype(np.bool)
        img_show = cv_image.copy()
        color_mask = color_masks[0]
        img_show[cur_mask_bool] = cv_image[cur_mask_bool]*0.5+color_mask*0.5
        cv2.imshow('mamamamsk', img_show)
        cv2.waitKey(0)
 
        if show_result == True:
            t_prev = time.time()
            solo_result = self.model.show_result(cv_image, result, score_thr=0.5)

            #display fps
            fps = int(1/(time.time()-t_prev))
            cv2.rectangle(solo_result, (5, 5), (75, 25), (0,0,0), -1)
            cv2.putText(solo_result, f'FPS {fps}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow("SOLOv2 Instance Segmentation Result", solo_result)
            cv2.waitKey(0)

            # # 按下q鍵退出程式
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     rospy.signal_shutdown("quit")


if __name__=="__main__":
    SOLO_Det()
