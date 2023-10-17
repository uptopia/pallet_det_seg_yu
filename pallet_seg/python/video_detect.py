#!/usr/bin/env python3

import cv2
import sys
sys.path.insert(1, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
Workspace_Path = sys.path[0].rsplit('/',2)[0]
sys.path.append(Workspace_Path)
sys.path.append(Workspace_Path+'/mmdetection2/')

from mmdet.apis import init_detector, inference_detector
import numpy as np
import os
import time

#=====Parameters Setting=====#
show_result = True
show_mask = True
save_result = False

ws_path = Workspace_Path #'/home/robotarm/forklift_pallet_ws/src'
test_img_path = ws_path + '/solo_detect/test_img/altek_img_1.jpg'
video_path = ws_path + '/pallet_seg/test_img/test.avi'
video_out_path = ws_path + '/pallet_seg/test_img/out.avi'


checkpoint_file = ws_path + '/mmdetection2/work_dirs/pallet.pth' #SOLO/configs/solov2/***.py
config_file = ws_path + '/mmdetection2/configs/solov2/pallet_test_2.py'                      #SOLO/data/***.pth
score_thr=0.5
#=====Parameters Setting=====#

if __name__ == '__main__':
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
    
    while cap.isOpened():
        ret, cv_image = cap.read()
        if not ret: break

        result = inference_detector(model, cv_image)
        
        t_prev = time.time()
        solo_result = model.show_result(cv_image, result, score_thr=score_thr)

        #display fps and Result
        fps = int(1/(time.time()-t_prev))
        cv2.rectangle(solo_result, (5, 5), (75, 25), (0,0,0), -1)
        cv2.putText(solo_result, f'FPS {fps}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("SOLOv2 Instance Segmentation Result", solo_result)
        # cv2.waitKey(1)
        
        seg_scores = result[0]
        # print(seg_scores)
        # print("=============")
        num_mask = len(result[0][0])
        elibible_mask=0
        for elibible_mask in range(num_mask):
            seg_scores = result[0][0][elibible_mask][4]
            if(seg_scores<=score_thr):
                break

        # print(elibible_mask)

        seg_box = result[1]
        num_box = len(seg_box)
        print("num_box:", num_box)
        mask_all = np.zeros((height, width))
        img_show = cv_image.copy()
        # box_id = 0
        # print("box_id:", box_id)
        # print(len(seg_box[box_id]), seg_box[box_id])
        # num_mask = len(seg_box[0])
        seg_masks = seg_box[0]
        print("num_mask", num_mask)

        np.random.seed(42)
        color_masks = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(num_mask)
        ]

        for mask_id in range(elibible_mask):
            print("mask_id:", mask_id)
            cur_mask = seg_masks[mask_id]

            h, w = len(cur_mask), len(cur_mask[0])
            print(h, w, cur_mask.shape)
            print(cur_mask)
            
            # cur_mask = (cur_mask>0.5).astype(np.uint8)
            # cur_mask_bool = cur_mask.astype(np.bool)
            cur_mask = np.array((cur_mask>0.5), dtype = np.uint8)
            cur_mask_bool = np.array(cur_mask, dtype = bool)
            
            ret0, mask_thr = cv2.threshold(cur_mask, 0, 255, cv2.THRESH_BINARY)

            color_mask = color_masks[mask_id]
            # r0, mask_thr = cv2.threshold(cur_mask_bool, 0, 255, cv2.THRESH_BINARY)
            img_show[cur_mask_bool] = color_mask#cv_image[cur_mask_bool]*0.5+color_mask*0.5
            mask_all+=mask_thr

            # Mask Result
            cv2.imshow('Black Background Mask', mask_all)
            cv2.imshow('Only Mask', img_show)
            # 按下q鍵退出程式
            if cv2.waitKey(1) == ord('q'):
                break

        out.write(solo_result)
    cv2.destroyAllWindows()
    cap.release()