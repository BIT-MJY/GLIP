from .generalized_rcnn import GeneralizedRCNN
from .generalized_vl_rcnn import GeneralizedVLRCNN

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN,
                                 "GeneralizedVLRCNN": GeneralizedVLRCNN  # swin_L用的是这个
                                 }


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
