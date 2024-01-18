# pallet_det_seg

![Ubuntu20.04](https://img.shields.io/badge/Ubuntu-20.04-green.svg)

| <div style="width:500px">托盤物件偵測 <br> Pallet object detection</div> | <div style="width:500px">托盤實例分割 <br> Pallet object detection</div> | 
| :----: | :----: | 
| <img src="readme_img/pallet_detect.jpg" width="320" height="180"/>  | <img src="readme_img/rgb_solo_mask2.png" width="320" height="180"/>  |

| <div style="width:500px">托盤點雲提取+姿態估測 <br> Pallet point cloud extraction & pose estimation</div> | <div style="width:500px">托盤點雲提取+姿態估測 <br> Pallet point cloud extraction & pose estimation</div> | 
| :----: | :----: | 
| <img src="readme_img/pallet_pose.png" width="320" height="180"/>  | <img src="readme_img/pose2.png" width="320" height="180"/>  |

## Table of content
- [RGB-D Camera](#rgb-d-camera)
- [Installation](#installation)
- [托盤物件偵測](#托盤物件偵測)
- [托盤實例分割](#托盤實例分割)

## RGB-D Camera
* [Realsense D435i](https://www.intelrealsense.com/depth-camera-d435i)
    - [ ] How to modify dockerfile
* [Altek 3D UVC](https://store.altek.com.tw/sites/default/downloads/al3d/altek_3D_UVC_Camera_Product_Specification_20220107.pdf)  
    * silver_camera: Altek_3D_Camera SDK (v2.49.0 tag:unknown)
    * black_camera: Altek_3D_Camera SDK (v2.49.0 tag:v67)
    - [ ] How to modify dockerfile

## Installation
```
mkdir pallet_det_seg_ws
git clone --recursive https://github.com/AndersonYu7/pallet_det_seg.git src
cd pallet_det_seg_ws/src/docker && ./build.sh

！！！相機先連接主機 再執行 不然會找不到相機
./run.sh
```
inside docker container
```
# install mmdetection
cd ~/work/src/mmdetection2
pip3 install -v -e .

# compile darknet (YOLOv4)
cd ~/work/src/darknet_new
make

# build project
cd ~/work
catkin_make
. devel/setup.bash

# USB port
sudo chmod 777 /dev/video0
```

## 托盤物件偵測
### Function Modules
| # | <div style="width:500px">Name</div> | rospkg name | Description |
| :----: | :----: | :----: | :----: |
| A1 | 托盤物件偵測 <br> Pallet object detection | pallet_det | Detect wood pallets bounding boxes from RGB image |
| B1 | 托盤點雲提取+姿態估測 <br> Pallet point cloud extraction <br>& pose estimation | pallet_det_pose | Extract 3D points relative to pallet bounding boxes' pixels from organized point cloud and estimate pallet's pose |
| A2 | 托盤物件偵測 <br> Pallet object detection | pallet_det | Detect wood pallets bounding boxes from RGB image |
| B2 | 托盤點雲提取+姿態估測 <br> Pallet point cloud extraction <br>& pose estimation | pallet_det_pose | Extract 3D points relative to pallet bounding boxes' pixels from organized point cloud and estimate pallet's pose |

### 下載pallet YOLOv4權重檔
* YOLOv4
* config:
* weights

```mermaid
---
title: Pallet Object Detection + Pose Estimation
---
%%{init: {"flowchart": {"wrappingWidth": 700, "htmlLabels": true}} }%%
graph LR 
    A[Pallet\nObject\nDetection] 
    A -->|ROS\nTopics| B(Realtime?)
    A -->|ROS\nServices\n施工中| C(Realtime?)
    B -->|N: rosbag | D("`__onlyA1__ roslaunch pallet_det pallet_det_**bag**.launch
                        __A1+B1__ roslaunch pallet_det_pose pallet_det_pose_**bag**.launch`")
    B -->|Y: camera | E("`__onlyA1__ roslaunch pallet_det pallet_det_**cam**.launch
                        __A1+B1__ roslaunch pallet_det_pose pallet_det_pose_**cam**.launch`")
    C -->|N: rosbag | F("`__onlyA2__ roslaunch pallet_det_**srv** pallet_det_**srv_bag**.launch
                        __A2+B2__ roslaunch pallet_det_pose_**srv** pallet_det_pose_**srv_bag**.launch`")
    C -->|Y: camera | G("`__onlyA2__ roslaunch pallet_det_**srv** pallet_det_**srv_cam**.launch
                        __A2+B2__ roslaunch pallet_det_pose_**srv** pallet_det_pose_**srv_cam**.launch`")

    style D text-align:left
    style E text-align:left
    style F text-align:left
    style G text-align:left 
```
Realtime?  * [N: rosbag] --> 使用預錄ROSBag, [Y: camera] --> 使用即時相機資訊

## 托盤實例分割
### Function Modules
| # | <div style="width:500px">Name</div> | rospkg name | Description |
| :----: | :----: | :----: | :----: |
| C1 | 托盤實例分割 <br> Pallet instance segmentation | pallet_seg | Detect and perform instance segmentation of wood pallets instance masks from RGB image |
| D1 | 托盤點雲提取+姿態估測 <br> Pallet point cloud extraction <br>& pose estimation | pallet_seg_pose | Extract 3D points relative to DLO instance mask pixels from organized point cloud and estimate pallet's pose |
| C2 | 托盤實例分割 <br> Pallet instance segmentation | pallet_seg | Detect and perform instance segmentation of wood pallets instance masks from RGB image |
| D2 | 托盤點雲提取+姿態估測 <br> Pallet point cloud extraction <br>& pose estimation | pallet_seg_pose | Extract 3D points relative to DLO instance mask pixels from organized point cloud and estimate pallet's pose |

### 下載pallet SOLOv2權重檔
* SOlOv2 with [mmdetection2](https://github.com/open-mmlab/mmdetection)
* config: ./mmdetection2/configs/solov2/pallet_test_2.py  
* weights: pallet.pth  
(把權重檔放在~/work/src/mmdetection2/work_dirs/pallet.pth)
```
# inside docker container
cd ~/work/src/mmdetection2/
mkdir work_dirs

# copy weight into docker container
docker cp pallet.pth 896ac4d402bb:/home/iclab/work/src/mmdetection2/work_dirs
```

### 程式執行
```mermaid
---
title: Pallet Instance Segmentation + Pose Estimation
---
%%{init: {"flowchart": {"wrappingWidth": 700, "htmlLabels": true, "defaultRenderer": "dagre-d3"}} }%%
graph LR
    A[Pallet\nInstance\nSegmentation] 
    A -->|ROS\nTopics| B(Realtime?)
    A -->|ROS\nServices\n施工中| C(Realtime?)
    B -->|N: rosbag | D("`__onlyC1__ roslaunch pallet_seg pallet_seg_**bag**.launch
                        __C1+D1__ roslaunch pallet_seg_pose pallet_seg_pose_**bag**.launch`")
    B -->|Y: camera | E("`__onlyC1__ roslaunch pallet_seg pallet_seg_**cam**.launch
                        __C1+D1__ roslaunch pallet_seg_pose pallet_seg_pose_**cam**.launch`")
    C -->|N: rosbag | F("`__onlyC2__ roslaunch pallet_seg_srv pallet_seg_**srv_bag**.launch
                        __C2+D2__ roslaunch pallet_seg_pose_srv pallet_seg_pose_**srv_bag**.launch`")
    C -->|Y: camera | G("`__onlyC2__ roslaunch pallet_seg_srv pallet_seg_**srv_cam**.launch
                        __C2+D2__ roslaunch pallet_seg_pose_srv pallet_seg_pose_**srv_cam**.launch`")

    style D text-align:left
    style E text-align:left
    style F text-align:left
    style G text-align:left 
```
Realtime?  * [N: rosbag] --> 使用預錄ROSBag, [Y: camera] --> 使用即時相機資訊