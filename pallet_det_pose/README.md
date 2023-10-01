# Object Detection -- YOLOv4
[YOLOv4 darknet]()
[YOLOv4 darknet_ros]()

## YOLOv4 Pallet Training
訓練模型與參數設定：
[launch] darknet_ros/launch/pallet_5.launch  
[yaml] darknet_ros/config/pallet_5.yaml  
[cfg] darknet_ros/yolo_network_config/cfg/pallet_5_yolov4.cfg
[weights] darknet_ros/yolo_network_config/weights/pallet_5_yolov4_final.weights

訓練指令：
`./darknet detector train data/obj.data yolo-obj.cfg yolov4.conv.137`  
`./darknet detector train data/obj.data yolov4-tiny-obj.cfg yolov4-tiny.conv.29`

./darknet detector train /home/robotarm/forklift_pallet_ws/src/train/new/01_02_tiny/cfg/pallet.data /home/robotarm/forklift_pallet_ws/src/train/new/01_02_tiny/cfg/pallet.cfg /home/robotarm/forklift_pallet_ws/src/train/new/01_02_tiny/cfg/yolov4-tiny.conv.29

obj.data you must specify the path to the validation dataset
.cfg 訓練網路與超參數


驗證指令：`./darknet detector valid cfg/coco.data cfg/yolov4.cfg yolov4.weights`

## YOLOv4 Pallet Testing
測試指令：`./darknet detector test ./cfg/coco.data ./cfg/yolov4.cfg ./yolov4.weights`

## YOLOv4 Pallet Pose Estimation
```
<terminal 1>
. devel/setup.bash
roslaunch darknet_ros pallet_5.launch

<terminal 2>
. devel/setup.bash
rosrun cloud_yolo cloud_yolo
```
darknet_ros/darknet_ros/test/test_main.cpp
/home/robotarm/altek_forklift_pallet_ws/src/darknet_ros/darknet_ros/test/test_main.cpp

## Sample Images
<img src="/readme_img/pallet_yolo.png" alt="drawing" width="800"/>

<table>
    <tr>
        <th>Lab</th><th></th><th colspan="3">Angle</th>
    </tr>
    <tr>
        <th></th><th></th><td align='center'>平視</td><td align='center'>高視角</td>
    </tr>
    <tr>
        <th rowspan="3" valgin='middle'>Light</td><td align='center'>正常</td><td><img src="/readme_img/pallet_solo.png" alt="drawing" width="800"/>  </td><td><img src="/readme_img/pallet_solo.png" alt="drawing" width="800"/>  </td>
    </tr>
    <tr>
        <td align='center'>昏暗</td><td><img src="/readme_img/pallet_solo.png" alt="drawing" width="800"/>  </td><td><img src="/readme_img/pallet_solo.png" alt="drawing" width="800"/>  </td>
    </tr>
    <tr>
        <td align='center'>強側光</td><td><img src="/readme_img/pallet_solo.png" alt="drawing" width="800"/>  </td><td><img src="/readme_img/pallet_solo.png" alt="drawing" width="800"/>  </td>
    </tr>
</table>

<table>
    <tr>
        <th>NotLab</th><th></th><th colspan="3">Angle</th>
    </tr>
    <tr>
        <th></th><th></th><td align='center'>平視</td><td align='center'>高視角</td>
    </tr>
    <tr>
        <th rowspan="3" valgin='middle'>Light</td><td align='center'>正常</td><td><img src="/readme_img/pallet_solo.png" alt="drawing" width="800"/>  </td><td><img src="/readme_img/pallet_solo.png" alt="drawing" width="800"/>  </td>
    </tr>
    <tr>
        <td align='center'>昏暗</td><td><img src="/readme_img/pallet_solo.png" alt="drawing" width="800"/>  </td><td><img src="/readme_img/pallet_solo.png" alt="drawing" width="800"/>  </td>
    </tr>
    <tr>
        <td align='center'>強側光</td><td><img src="/readme_img/pallet_solo.png" alt="drawing" width="800"/>  </td><td><img src="/readme_img/pallet_solo.png" alt="drawing" width="800"/>  </td>
    </tr>
</table>