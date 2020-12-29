from yacs.config import CfgNode as CN

_C = CN()

# Model info for GPA2Cls v1
_C.MODEL = CN()
_C.MODEL.NAME = "GPA2Cls-V1-Base"
_C.MODEL.INTRO = "The next for SEGPM2Cls-V0b"

# Backbone Network
_C.BACKBONE = CN()
_C.BACKBONE.NAME = "BACKBONE"
# Path or type for backbone network pretrained weights
_C.BACKBONE.PRETRAINED = False
_C.BACKBONE.PRETRAINED_FILE = None
_C.BACKBONE.STAGES = None

# Object Focal Locator
_C.FOCAL_LOCATOR = CN()
_C.FOCAL_LOCATOR.NAME = "FL"
_C.FOCAL_LOCATOR.SIZE = 448
_C.FOCAL_LOCATOR.STRIDE = 1
_C.FOCAL_LOCATOR.FOCAL_SIZE = 448

# Scaling Layer
_C.SCALING_LAYER = CN()
_C.SCALING_LAYER.NAME = "SCALE"
_C.SCALING_LAYER.IN_CHANNELS = None
_C.SCALING_LAYER.OUT_CHANNELS = None
_C.SCALING_LAYER.TRANSPARENT = True

# Global Perception Attention Layer
_C.GLOBAL_PERCEPTION_ATTN_LAYER = CN()
_C.GLOBAL_PERCEPTION_ATTN_LAYER.NAME = "GPA"
_C.GLOBAL_PERCEPTION_ATTN_LAYER.IN_CHANNELS = None
_C.GLOBAL_PERCEPTION_ATTN_LAYER.SPLIT_SIZES = None
_C.GLOBAL_PERCEPTION_ATTN_LAYER.SE_DESIGNS = None
_C.GLOBAL_PERCEPTION_ATTN_LAYER.TRANSPARENT = True

# Classifier
_C.CLASSIFIER = CN()
_C.CLASSIFIER.NAME = "CLASSIFIER"
_C.CLASSIFIER.IN_FEATURES = None
_C.CLASSIFIER.NUM_CLASSES = 0
_C.CLASSIFIER.POOLING = None
_C.CLASSIFIER.DROPOUT = False


def get_cfg_defaults():
  return _C.clone()

if __name__ == "__main__":
  cfg = get_cfg_defaults()
  cfg.merge_from_file("gpa2cls-v1.yaml")
  cfg.freeze()
  print(cfg)
