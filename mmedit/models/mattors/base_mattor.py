# Copyright (c) OpenMMLab. All rights reserved.
import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import ConfigDict
from mmengine.logging import MMLogger
from mmengine.model import BaseModel
from mmengine.config import Config
from mmengine.model import ImgDataPreprocessor, BaseDataPreprocessor
from mmedit.data_element import EditDataSample, PixelData
# from mmedit.core.evaluation import connectivity, gradient_error, mse, sad
# from ..base import BaseModel
# from ..builder import build_backbone, build_component
from mmedit.registry import MODELS
from mmengine.utils import stack_batch
# 2022-02-20 22:26:38,860 - mmdet - INFO - this is a test

DataSamples = Optional[Union[list, torch.Tensor]]
ForwardResults = Union[Dict[str, torch.Tensor], List[EditDataSample],
                       Tuple[torch.Tensor], torch.Tensor]


# class TrimapBasedMattor(BaseModel, metaclass=ABCMeta):
class BaseMattor(BaseModel, metaclass=ABCMeta):
    """Base class for matting model.

    A matting model must contain a backbone which produces `alpha`, a dense
    prediction with the same height and width of input image. In some cases,
    the model will has a refiner which refines the prediction of the backbone.

    The subclasses should overwrite the function ``forward_train`` and
    ``forward_test`` which define the output of the model and maybe the
    connection between the backbone and the refiner.

    Args:
        backbone (dict): Config of backbone.
        refiner (dict): Config of refiner.
        train_cfg (dict): Config of training. In ``train_cfg``,
            ``train_backbone`` should be specified. If the model has a refiner,
            ``train_refiner`` should be specified.
        test_cfg (dict): Config of testing. In ``test_cfg``, If the model has a
            refiner, ``train_refiner`` should be specified.
        pretrained (str): Path of pretrained model.
    """

    def __init__(
        self,
        data_preprocessor: Union[dict, Config],
        backbone: dict,
        # refiner=None,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        pretrained=None,
    ):
        # Initialize nn.Module
        # Build data processor
        super().__init__(data_preprocessor=data_preprocessor)

        self.train_cfg = train_cfg if train_cfg is not None else ConfigDict()
        self.test_cfg = test_cfg if test_cfg is not None else ConfigDict()

        self.backbone = MODELS.build(backbone)

        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the model with pretrained weights.

        Args:
            pretrained (str, optional): Path to the pretrained weight.
                Defaults to None.
        """
        if pretrained is not None:
            logger = MMLogger.get_instance(name='mmedit')
            logger.warn(f'load model from: {pretrained}')
            self.backbone.init_weights(pretrained)
        # if self.with_refiner:
        #     self.refiner.init_weights()

    def restore_shape(self, pred_alpha, data_sample):
        """Restore the predicted alpha to the original shape.

        The shape of the predicted alpha may not be the same as the shape of
        original input image. This function restores the shape of the predicted
        alpha.

        Args:
            pred_alpha (np.ndarray shape=(1,H,W)): a single predicted alpha.
            meta (list[dict]): Meta data about the current data batch.
                Currently only batch_size 1 is supported.

        Returns:
            np.ndarray: The reshaped predicted alpha.
        """

        raise NotImplementedError
        pred_alpha = pred_alpha.squeeze()
        # print(pred_alpha.sum(), pred_alpha.max(), pred_alpha.min())
        ori_trimap = data_sample.ori_trimap.squeeze()
        ori_h, ori_w = data_sample.ori_merged_shape[:2]

        assert pred_alpha.ndim == ori_trimap.ndim == 2
        assert ori_trimap.shape == (ori_h, ori_w)

        if hasattr(data_sample, 'interpolation'):
            # images have been resized for inference, resize back
            raise NotImplementedError
            pred_alpha = mmcv.imresize(
                pred_alpha, (ori_w, ori_h),
                interpolation=meta[0]['interpolation'])
        elif hasattr(data_sample, 'pad_width'):
            # images have been padded for inference, remove the padding
            # note; padding is applied only at the end
            pred_alpha = pred_alpha[:ori_h, :ori_w]

        assert pred_alpha.shape == (ori_h, ori_w)
        # print(pred_alpha.sum())
        # some methods do not have an activation layer after the last conv,
        # clip to make sure pred_alpha range from 0 to 1.
        # pred_alpha = np.clip(pred_alpha, 0, 1)
        # pred_alpha[ori_trimap == 0] = 0
        # pred_alpha[ori_trimap == 255] = 255

        # pred_alpha = np.round(pred_alpha * 255).astype(np.uint8)
        # print(pred_alpha)
        # print(pred_alpha.dtype)
        return pred_alpha

    def postprocess(
        self,
        # inputs: torch.Tensor,  # N, 4, H, W, float32
        pred_alpha: torch.Tensor,  # N, 1, H, W, float32
        data_samples: List[EditDataSample],
    ) -> List[EditDataSample]:
        """Post-processing for alpha predictions.

        1. Restore padded shape
        1. Mask with trimap
        1. clamp to 0-1
        1. to uint8
        """

        raise NotImplementedError

    def forward(self,
                batch_inputs: torch.Tensor,
                data_samples: DataSamples = None,
                mode: str = 'feat') -> List[EditDataSample]:
        """Forward function in test mode.

        Args:
            merged (Tensor): Image to predict alpha matte.
            trimap (Tensor): Trimap of the input image.
            meta (list[dict]): Meta data about the current data batch.
                Defaults to None.
            alpha (Tensor, optional): Ground-truth alpha matte.
                Defaults to None.
            test_mode (bool, optional): Whether in test mode. If ``True``, it
                will call ``forward_test`` of the model. Otherwise, it will
                call ``forward_train`` of the model. Defaults to False.

        Returns:
            List[EditDataElement]:
                Sequence of predictions packed into EditDataElement
        """

        if mode == 'feat':
            raw = self._forward(batch_inputs, data_samples)
            return raw
        elif mode == 'predict':
            # Pre-process runs in runner
            pred_alpha = self._forward_test(batch_inputs, data_samples)
            predictions = self.data_preprocessor.postprocess(
                pred_alpha, data_samples)
            return predictions
        elif mode == 'loss':
            loss = self._forward_train(batch_inputs, data_samples)
            return loss

    def forward_dummy(self, inputs):
        """For ONNX."""
        return self._forward(inputs, self.with_refiner)