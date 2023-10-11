# pallet_det_seg
## SOlOv2 with mmdetection2
### https://github.com/open-mmlab/mmdetection

config: ./mmdetection2/configs/solov2/pallet_test_2.py

weights: pallet.pth

# Main
1. git clone --recursive https://github.com/AndersonYu7/pallet_det_seg.git
2. cd docker && ./build.sh
3. cd mmdetection2
4. pip3 install -v -e .
5. cd ../darknet_new
6. make
7. cd ../..
8. catkin_make
9. . devel/setup.bash

## 執行seg realtime
roslaunch pallet_seg pallet_seg_altek_cam.launch

## 執行seg&pose realtime
roslaunch pallet_seg_pose solo_pallet_cam.launch
