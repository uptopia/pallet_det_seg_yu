#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@file pallet_seg_srv.py
@brief 即時顯示當前相機影像&SOLOv2預測結果，
       將SOLOv2遮罩publish到/solo_mask topic上

參數設定：
(1) show_result = True  #即時顯示RGB影像及SOLOv2遮罩
(2) 
(3) save_result = False #儲存RGB影像及SOLOv2遮罩
    影像儲存位置：./src/solo_detect/test_img/solo_res_{}.jpg
(4) use_test_img = False#使用test_img
(5) config_file         #SOLOv2訓練設定檔
(6) checkpoint_file     #SOLOv2權重檔
'''

import os
import sys
import time
import numpy as np

# OpenCV
import cv2
sys.path.insert(1, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
from cv_bridge import CvBridge, CvBridgeError

# mmdetection, mmcv
Workspace_Path = sys.path[0].rsplit('/',2)[0]
sys.path.append(Workspace_Path)
sys.path.append(Workspace_Path+'/mmdetection2/')
import mmcv
from mmdet.apis import init_detector, inference_detector

# ROS
import rospy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float64MultiArray, Int32MultiArray
import message_filters # TimeSynchronizer

# self-defined ROS msg, srv
from pallet_seg.msg import pallet_mask
from pallet_srv.srv import PalletSeg, PalletSegResponse

#=====Parameters Setting=====#
show_solo_result = True


ws_path = Workspace_Path #'/home/robotarm/forklift_pallet_ws/src'
test_img_path = ws_path + '/solo_detect/test_img/altek_img_1.jpg'

checkpoint_file = ws_path + '/mmdetection2/work_dirs/pallet.pth' #SOLO/configs/solov2/***.py
config_file = ws_path + '/mmdetection2/configs/solov2/pallet_test_2.py'                      #SOLO/data/***.pth
score_thr = 0.5
#=====Parameters Setting=====#

cnt = 0

class PalletSegServer:
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.rgb_img_msg = Image()
        self.organized_cloud_msg = PointCloud2()
        # self.pallet_masks_list = PalletCloudResponse.pallet_masks_list()
        self.pallet_masks_list = []

        rospy.init_node("PalletSegServer")

        # Build SOLOv2 model from a config file and a checkpoint file
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')
        
        # Synchronize & Subscribe topics
        rgb_sub   = message_filters.Subscriber("/camera/color/image_raw", Image)
        cloud_sub = message_filters.Subscriber("/camera/depth_registered/points", PointCloud2)

        syns = message_filters.ApproximateTimeSynchronizer([rgb_sub, cloud_sub], 10, 0.1, True)
        syns.registerCallback(self.multi_callback)

        # ROSService Server
        server = rospy.Service("/pallet_seg_service", PalletSeg, self.pallet_cloud_callback)

        self.pallet_mask_pub = rospy.Publisher("/pallet_mask", pallet_mask, queue_size=10)

        rospy.spin()

    def multi_callback(self, rgb_msg, cloud_msg):
        self.rgb_img_msg = rgb_msg
        self.organized_cloud_msg = cloud_msg
        #OK
        self.pallet_masks_list = self.pallet_ins_segmentation(rgb_msg)
        
    def pallet_cloud_callback(self, req):
        print("=============req:", req)
    
        if req.ready_to_get_pallet_cloud == True:

            # #not OK
            # curr_cloud = self.organized_cloud_msg
            # curr_rgb = self.rgb_img_msg

            # res = PalletSegResponse()
            # res.pallet_masks_list = self.pallet_ins_segmentation(curr_rgb)
            # res.organized_cloud_msg = self.organized_cloud_msg
            # return res

            #OK
            res = PalletSegResponse()
            # res.pallet_ids      = TODO
            # res.pallet_scores   = TODO
            res.pallet_masks_list = self.pallet_masks_list
            res.organized_cloud_msg = self.organized_cloud_msg
            return res
        else:
            print("req.ready_to_get_pallet_cloud == False")
            return None

    def pallet_ins_segmentation(self, rgb_img_msg):

        print("*******RUN pallet_ins_segmentation************")
        # ============
        #  RGB Image 
        # ============
        cv_image = self.bridge.imgmsg_to_cv2(self.rgb_img_msg, "bgr8")
        # cv_image = cv2.imread("/home/iclab/work/src/pallet_seg/test_img/altek_img_1.jpg")
        h, w, _ = cv_image.shape

        # =============================
        # SOLOv2 Instance Segmentation
        # =============================
        result = inference_detector(self.model, cv_image)

        t_prev = time.time()
        solo_result = self.model.show_result(cv_image, result, score_thr=score_thr)

        # Display fps and Result
        fps = int(1/(time.time()-t_prev))
        cv2.rectangle(solo_result, (5, 5), (75, 25), (0,0,0), -1)
        cv2.putText(solo_result, f'FPS {fps}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("SOLOv2 Instance Segmentation Result (score, bbox, mask)", solo_result)
        # cv2.waitKey(1)

        # == Instance Masks ==
        num_mask = len(result[0][0])
        elibible_mask=0
        for elibible_mask in range(num_mask):
            seg_scores = result[0][0][elibible_mask][4]
            if(seg_scores<=score_thr):
                break

        # == Bounding Boxes ==
        seg_box = result[1]
        num_box = len(seg_box)
        print("num_box:", num_box)

        mask_all = np.zeros((h, w))
        img_show = cv_image.copy()
        
        # box_id = 0
        # print("box_id:", box_id, len(seg_box[box_id]), seg_box[box_id])
        # num_mask = len(seg_box[0])
        seg_masks = seg_box[0]
        print("num_mask", num_mask)

        # random mask color
        np.random.seed(42)
        color_masks = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(num_mask)
        ]

        pallet_mask_msg = []
        for mask_id in range(elibible_mask):
            print("mask_id:", mask_id)
            cur_mask = seg_masks[mask_id]

            h, w = len(cur_mask), len(cur_mask[0])
            # print(h, w, cur_mask.shape)
            # print(cur_mask)
            
            cur_mask = np.array((cur_mask>0.5), dtype = np.uint8)
            cur_mask_bool = np.array(cur_mask, dtype = bool)
            
            ret0, mask_thr = cv2.threshold(cur_mask, 0, 255, cv2.THRESH_BINARY)

            color_mask = color_masks[mask_id]
            # r0, mask_thr = cv2.threshold(cur_mask_bool, 0, 255, cv2.THRESH_BINARY)
            img_show[cur_mask_bool] = color_mask #cv_image[cur_mask_bool]*0.5+color_mask*0.5
            mask_all+=mask_thr

            #pub         
            mask_msg = self.bridge.cv2_to_imgmsg(mask_thr, encoding="passthrough")
            pallet_mask_msg.append(mask_msg)

        # # service
        # self.pallet_masks_list = pallet_mask_msg

        # Mask Result
        if show_solo_result == True:
            cv2.imshow('Pallet Masks_Black Background', mask_all)
            cv2.imshow('Pallet Masks_RGB image + Masks', img_show)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press q to exit
                rospy.signal_shutdown("quit")
        #pub
        self.pallet_mask_pub.publish(pallet_mask_msg)
        return pallet_mask_msg

if __name__=="__main__":
    PalletSegServer()