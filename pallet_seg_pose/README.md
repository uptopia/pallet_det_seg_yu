# Instance Segmentation -- SOLOv2
[SOLOv2](https://github.com/WXinlong/SOLO)

## SOLOv2 Pallet Training
訓練模型與參數設定：

訓練指令：
```
# multi-gpu training
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}

Example: 
./tools/dist_train.sh configs/solo/solo_r50_fpn_8gpu_1x.py  8

# single-gpu training
python tools/train.py ${CONFIG_FILE}

Example: 
python tools/train.py configs/solo/solo_r50_fpn_8gpu_1x.py
```
./tools/dist_train.sh configs/solov2/solov2_light_448_r18_fpn_8gpu_3x.py 4
python3 tools/train.py configs/solov2/solov2_light_448_r18_fpn_8gpu_3x.py
/home/robotarm/SOLO/data/Pallet_dataset/test3/latest.pth

驗證指令：

## SOLOv2 Pallet Testing
測試指令：
```
# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}  --show --out  ${OUTPUT_FILE} --eval segm

Example: 
./tools/dist_test.sh configs/solo/solo_r50_fpn_8gpu_1x.py SOLO_R50_1x.pth  8  --show --out results_solo.pkl --eval segm

# single-gpu testing
python tools/test_ins.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show --out  ${OUTPUT_FILE} --eval segm

Example: 
python tools/test_ins.py configs/solo/solo_r50_fpn_8gpu_1x.py  SOLO_R50_1x.pth --show --out  results_solo.pkl --eval segm
```

```
python tools/test_ins_vis.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show --save_dir  ${SAVE_DIR}

Example: 
python tools/test_ins_vis.py configs/solo/solo_r50_fpn_8gpu_1x.py  SOLO_R50_1x.pth --show --save_dir  work_dirs/vis_solo
```

python3 demo/inference_demo.py
python3 tools/test_ins_vis.py configs/solov2/solov2_light_448_r18_fpn_8gpu_3x.py data/Pallet_dataset/solov2_light_release_r18_fpn_8gpu_3x/latest.pth --show --save_dir work_dirs/vis_solo
python3 tools/test_ins_vis.py configs/solov2/solov2_light_448_r18_fpn_8gpu_3x.py /home/robotarm/SOLO/data/Pallet_dataset/test/latest.pth --show --save_dir /home/robotarm/SOLO/data/Pallet_dataset/vis_solo

---
### 使用rosbag
rosbag play -l one_pallet.bagaa
. devel/setup.bash
roslaunch solo_cloud solo_pallet.launch

## SOLOv2 Pallet Pose Estimation
```
<terminal 1>
. devel/setup.bash
roslaunch solo_cloud solo_pallet.launch

<terminal 2>
. devel/setup.bash
rosrun solo_cloud solo_cloud
```
## Sample Images
<img src="/readme_img/pallet_solo.png" alt="drawing" width="800"/>  

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