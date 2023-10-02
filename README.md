# pallet_det_seg
## SOlOv2 with mmdetection2
### https://github.com/open-mmlab/mmdetection

config: ./mmdetection2/configs/solov2/pallet_test_2.py

weights: pallet.pth

1. cd docker && ./build
2. cd mmdetection2
3. pip3 install -v -e .
4. cd ../..
5. catkin_make
6. . devel/setup.bash

## 執行seg realtime
roslaunce realsense2_camera rs_camera.launch

rosrun pallet_seg solo_detect.py
