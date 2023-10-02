_base_ = 'solov2_light_r18_fpn_3x_coco.py'

import os
import sys
import torch
mmdetection_path = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))+'/mmdetection2'
sys.path.append(mmdetection_path)
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#set PYTOURCH_CUDA_ALLOC_CONF = max_split_size_mb:6144
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
# torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32 "


dataset_type = 'PalletDataset'
data_root = mmdetection_path + '/data/pallet_coco/'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root = data_root,
        ann_file=data_root+'pallet_train/pallet_train_annotations.json',
        img_prefix=data_root+'pallet_train/'),
    val=dict(
        type=dataset_type,
        data_root = data_root,
        ann_file=data_root+'pallet_val/pallet_val_annotations.json',
        img_prefix=data_root+'pallet_val/'),
    test=dict(
        type=dataset_type,
        data_root = data_root,
        ann_file=data_root+'pallet_val/pallet_test_annotations.json',
        img_prefix=data_root+'pallet_test/'))

model = dict(
    mask_head=dict(num_classes=2)
)
