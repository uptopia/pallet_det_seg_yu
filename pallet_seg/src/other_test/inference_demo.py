from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import cv2


show_res = True
save_result = False
config_file = '/home/robotarm/solo_cam_ws/src/pallet_seg/src/SOLO/configs/solov2/solov2_light_448_r18_fpn_8gpu_3x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '/home/robotarm/solo_cam_ws/src/pallet_seg/src/SOLO/data/pallet_1/solov2_light_release_r18_fpn_8gpu_3x/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
for i in range(0,5):
    img = f'/home/robotarm/forklift_pallet_ws/src/pallet_seg/img/altek_img_{i}.jpg'
    result = inference_detector(model, img)

    solo_result = show_result_ins_bbox(img, result, model.CLASSES, score_thr=0.25)#, out_file=f"demo_out__{i}.jpg")
    cv2.imshow("img_show", solo_result)
    cv2.waitKey(0)