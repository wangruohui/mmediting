# Copyright (c) OpenMMLab. All rights reserved.
from .cli import modify_args
from .logger import get_root_logger
from .setup_env import register_all_modules, setup_multi_processes

__all__ = [
    'modify_args',
    'get_root_logger',
    'register_all_modules',
    'setup_multi_processes',
]
