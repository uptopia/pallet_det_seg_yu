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
from pallet_srv.srv import PalletCloud

class PalletCloudClient:
    def __init__(self, ready_to_get_pallet_cloud:True) -> None:
        self.get_pallet_cloud = rospy.ServiceProxy('/pallet_seg_service', PalletCloud)
        self.ready_to_get_pallet_cloud = ready_to_get_pallet_cloud
        self.usage()
        while True:
            self.get_pallet_cloud_client()
    
    def usage(self):
        print("Requesting [ready_to_get_pallet_cloud] = %s"%(str(self.ready_to_get_pallet_cloud)))
        
    def get_pallet_cloud_client(self):
        rospy.wait_for_service('/pallet_seg_service')
        try:
            res = self.get_pallet_cloud(self.ready_to_get_pallet_cloud)

            print("Server Response...")
            # print("pallet_masks_list: ", res.pallet_masks_list)
            # print("organized_cloud_msg: ", res.organized_cloud_msg)
        except rospy.ServiceException as e:
            print("Service call failed: ", e)

if __name__ == "__main__":
    PalletCloudClient(True)
    # if len(sys.argv) == 2:
    #     PalletCloudClient(sys.argv[1])
    # else:
    #     print("Error: require 1 arguement [ready_to_get_pallet_cloud = True/False] to execute")
    #     sys.exit(1)