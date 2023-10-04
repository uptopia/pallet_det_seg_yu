# pallet_det_seg
## SOlOv2 with mmdetection2
### https://github.com/open-mmlab/mmdetection

config: ./mmdetection2/configs/solov2/pallet_test_2.py

weights: pallet.pth

1. git clone --recursive 
2. cd docker && ./build
3. cd mmdetection2
4. pip3 install -v -e .
5. cd ../..
6. catkin_make
7. . devel/setup.bash

## 執行seg realtime
roslaunce realsense2_camera rs_camera.launch

rosrun pallet_seg solo_detect.py
