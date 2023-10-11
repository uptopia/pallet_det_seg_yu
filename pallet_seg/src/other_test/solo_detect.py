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
        # self.solo_mask_pub = rospy.Publisher("/solo_mask", Image, queue_size=10)
        rospy.spin()

    def imageCallback(self, img_msg):
        global cnt

        cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        result = inference_detector(self.model, cv_image)
        cv2.imshow("Result", cv_image)
        # cv2.waitKey(1)

        # if show_result == True:

            # solo_result = show_result_ins(cv_image, result, self.model.CLASSES, score_thr=0.25)#, out_file=f"/home/robotarm/solo_cam_ws/src/solo_detect/img/demo_out_{cnt}.jpg")
            # solo_result = show_result_pyplot(self.model, cv_image, result)
        solo_result = self.model.show_result(
            cv_image,
            result,
            score_thr=0.25)
        cv2.imshow("SOLOv2 Instance Segmentation Result", solo_result)
        cv2.waitKey(1)

            # if save_result == True:
            #     cv2.imwrite(ws_path + "/solo_detect/test_img/solo_res_{}.jpg".format(cnt), solo_result)
            #     print("save img: solo_res_{}.jpg".format(cnt))
            #     cnt+=1

            # #========================#
            # # View+Publish solo_mask
            # #========================#
            # num_mask, h, w = seg_label.shape
            # # print("num_mask, h, w", num_mask, h, w)
            # mask_all = np.zeros((h,w))  

            # if show_mask == True:
                
            #     #=== Show all the masks in ONE window ===#
            #     for mask in seg_label:
            #         mask_all += mask
            #     ret, mask_all_thr = cv2.threshold(mask_all, 0, 255, cv2.THRESH_BINARY)
            #     cv2.imshow("SOLOv2 Mask Result", mask_all_thr)
            #     cv2.waitKey(1)

            #     # #=== Show masks in SEPARATE windows ===#
            #     # for k in range(num_mask):
            #     #     ret, th1 = cv2.threshold(seg_label[k], 0, 255, cv2.THRESH_BINARY)
            #     #     cv2.imshow("seg_label"+str(k), th1)
            #     #     cv2.waitKey(1)

            # solo_mask_msg = self.bridge.cv2_to_imgmsg(mask_all_thr, encoding="passthrough")
            # self.solo_mask_pub.publish(solo_mask_msg)
            
            # #https://blog.csdn.net/weixin_43056273/article/details/120571808
            # print(solo_mask_msg.header)
            # print(solo_mask_msg.height)       #360
            # print(solo_mask_msg.width)        #640
            # print(solo_mask_msg.encoding)     #8UC1
            # print(solo_mask_msg.is_bigendian) #0
            # print(solo_mask_msg.step)         #640
            # print(solo_mask_msg.data)

if __name__=="__main__":
    SOLO_Det()
