from .coco import CocoDataset
from .builder import DATASETS

@DATASETS.register_module()
class PalletDataset(CocoDataset):
    CLASSES = ('null', 'front')
    PALETTE = ((220, 20, 60), (0, 0, 0))