#! /usr/bin/env python3

import sys
import rospy
import numpy as np
from pallet_srv.srv import PalletPose

class PalletCloudClient:
    def __init__(self, arrived_at_take_pic_pose:True) -> None:
        self.usage()
        self.get_pallet_pose_client(arrived_take_pic_pose, forklift_H_cam)

    def usage(self):
        print("Requesting [arrived_take_pic_pose] = %s"%(str(self.arrived_take_pic_pose)))
        print("forklift_H_cam = ", forklift_H_cam)

    def get_pallet_pose_client(self, arrived_take_pic_pose, forklift_H_cam):
        rospy.wait_for_service('/pallet_pose_service')
        try:
            get_pallet_pose = rospy.ServiceProxy('/pallet_pose_service', PalletPose)
            res = get_pallet_pose(arrived_take_pic_pose, forklift_H_cam)
            print(res.forklift_H_pallet)

            #TODO: plan trajectory to fork the pallet
            return res
        except rospy.ServiceException as e:
            print("Service call failed: ", e)

if __name__ == "__main__":

    arrived_take_pic_pose = True
    forklift_H_cam = np.identity(4) #TODO: get transformation

    PalletCloudClient(arrived_take_pic_pose, forklift_H_cam)