# Copyright (c) OpenMMLab. All rights reserved.
from .base_mattor import BaseMattor
from .data_processor import ImageAndTrimapPreprocessor
from .dim import DIM
from .gca import GCA
from .indexnet import IndexNet
from .plain_refiner import PlainRefiner

# from .utils import get_unknown_tensor

__all__ = [
    'BaseMattor',
    'ImageAndTrimapPreprocessor',
    'DIM',
    'PlainRefiner',
    'IndexNet',
    'GCA',
    'get_unknown_tensor',
]
