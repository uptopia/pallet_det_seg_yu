from .coco import CocoDataset
from .registry import DATASETS

@DATASETS.register_module
class PalletDataset(CocoDataset):
    CLASSES = ['null', 'front']
