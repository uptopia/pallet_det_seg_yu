#! /usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@file pallet_seg_srv.py
@brief 測試pallet_seg_service

參數設定：
(1) 
(2) 
'''

import sys
import numpy as np

# ROS
import rospy

# self-defined ROS msg, srv
from pallet_srv.srv import PalletSeg

class PalletSegClient:
    def __init__(self, ready_to_get_pallet_seg:True) -> None:
        self.get_pallet_seg = rospy.ServiceProxy('/pallet_seg_service', PalletSeg)
        self.ready_to_get_pallet_seg = ready_to_get_pallet_seg
        self.usage()
        self.get_pallet_seg_client()
    
    def usage(self):
        print("Requesting [ready_to_get_pallet_seg] = %s"%(str(self.ready_to_get_pallet_seg)))
        
    def get_pallet_seg_client(self):
        rospy.wait_for_service('/pallet_seg_service')
        try:
            res = self.get_pallet_seg(self.ready_to_get_pallet_seg)

            print("Server Response...")
            print("pallet_masks_list: ", len(res.pallet_masks_list))
            # print("organized_cloud_msg: ", res.organized_cloud_msg)
        except rospy.ServiceException as e:
            print("Service call failed: ", e)

if __name__ == "__main__":
    PalletSegClient(True)
    # if len(sys.argv) == 2:
    #     PalletSegClient(sys.argv[1])
    # else:
    #     print("Error: require 1 arguement [ready_to_get_pallet_seg = True/False] to execute")
    #     sys.exit(1)