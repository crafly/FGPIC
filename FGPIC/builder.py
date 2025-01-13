# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmcv.utils import ConfigDict, print_log
from mmdet.models.builder import DETECTORS

from mmdet.models.builder import MODELS
# from mmcv.cnn import MODELS as MMCV_MODELS
# from mmcv.utils import Registry
INFORMATION_COUPLING = MODELS
PROTOTYPE_AGGREGATION = MODELS


def build_information_coupling(cfg):
    """Build attention."""
    return INFORMATION_COUPLING.build(cfg)


def build_prototype_aggregation(cfg):
    """Build attention."""
    return PROTOTYPE_AGGREGATION.build(cfg)
