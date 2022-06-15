# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv import ConfigDict
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.model import BaseModel

from mmedit.data_element import EditDataSample, PixelData
from mmedit.registry import MODELS

DataSamples = Optional[Union[list, torch.Tensor]]
ForwardResults = Union[Dict[str, torch.Tensor], List[EditDataSample],
                       Tuple[torch.Tensor], torch.Tensor]


def _pad(batch_image, ds_factor, mode='reflect'):
    """Pad image to a multiple of give down-sampling factor."""

    h, w = batch_image.shape[-2:]  # NCHW

    new_h = ds_factor * ((h - 1) // ds_factor + 1)
    new_w = ds_factor * ((w - 1) // ds_factor + 1)

    pad_h = new_h - h
    pad_w = new_w - w
    pad = (pad_h, pad_w)
    if new_h != h or new_w != w:
        pad_width = (0, pad_w, 0, pad_h)  # torch.pad in reverse order
        batch_image = F.pad(batch_image, pad_width, mode)

    return batch_image, pad


def _interpolate(batch_image, ds_factor, mode='bicubic'):
    """Resize image to multiple of give down-sampling factor."""

    h, w = batch_image.shape[-2:]  # NCHW

    new_h = h - (h % ds_factor)
    new_w = w - (w % ds_factor)

    size = (new_h, new_w)
    if new_h != h or new_w != w:
        batch_image = F.interpolate(batch_image, size=size, mode=mode)

    return batch_image, size


class BaseMattor(BaseModel, metaclass=ABCMeta):
    """Base class for trimap-based matting models.

    A matting model must contain a backbone which produces `pred_alpha`,
    a dense prediction with the same height and width of input image.
    In some cases (such as DIM), the model has a refiner which refines
    the prediction of the backbone.

    Subclasses should overwrite the following functions:

    - :meth:`_forward_train`, to return a loss
    - :meth:`_forward_test`, to return a prediction
    - :meth:`_forward`, to return raw tensors

    For test, this base class provides functions to resize inputs and
    post-process pred_alphas to get predictions

    Args:
        backbone (dict): Config of backbone.
        data_preprocessor (dict): Config of data_preprocessor.
            See :class:`MattorPreprocessor` for details.
        train_cfg (dict): Config of training.
            Customized by subclassesCustomized bu In ``train_cfg``,
            ``train_backbone`` should be specified. If the model has a refiner,
            ``train_refiner`` should be specified.
        test_cfg (dict): Config of testing.
            In ``test_cfg``, If the model has a
            refiner, ``train_refiner`` should be specified.
        pretrained (str): Path of pretrained model.
    """

    def __init__(self,
                 data_preprocessor: Union[dict, Config],
                 backbone: dict,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 pretrained=None):
        # Initialize nn.Module
        # Build data processor
        super().__init__(data_preprocessor=data_preprocessor)

        self.train_cfg = train_cfg if train_cfg is not None else ConfigDict()
        self.test_cfg = test_cfg if test_cfg is not None else ConfigDict()

        self.backbone = MODELS.build(backbone)

        # sub-class should re-init if there are more modules
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the model with pretrained weights.

        Args:
            pretrained (str, optional): Path to the pretrained weight.
                Defaults to None.
        """
        if pretrained is not None:
            logger = MMLogger.get_instance(name='mmedit')
            logger.warn(f'Loading model from: {pretrained} ...')
        self.backbone.init_weights(pretrained)

    def resize_inputs(self, batch_inputs):
        """Pad or interpolate images and trimaps to multiple of given factor.
        """

        resize_method = self.test_cfg['resize_method']
        resize_mode = self.test_cfg['resize_mode']
        size_divisor = self.test_cfg['size_divisor']

        batch_images = batch_inputs[:, :3, :, :]
        batch_trimaps = batch_inputs[:, 3:, :, :]

        if resize_method == 'pad':
            batch_images, _ = _pad(batch_images, size_divisor, resize_mode)
            batch_trimaps, _ = _pad(batch_trimaps, size_divisor, resize_mode)
        elif resize_method == 'interp':
            batch_images, _ = _interpolate(batch_images, size_divisor,
                                           resize_mode)
            batch_trimaps, _ = _interpolate(batch_trimaps, size_divisor,
                                            'nearest')
        else:
            raise NotImplementedError

        return torch.cat((batch_images, batch_trimaps), dim=1)

    def restore_size(self, pred_alpha, data_sample):
        """Restore the predicted alpha to the original shape.

        The shape of the predicted alpha may not be the same as the shape of
        original input image. This function restores the shape of the predicted
        alpha.

        Args:
            pred_alpha (torch.Tensor): A single predicted alpha of
                shape (1, H, W).
            data_sample (EditDataSample): Data sample containing
                original shape as meta data.

        Returns:
            torch.Tensor: The reshaped predicted alpha.
        """
        resize_method = self.test_cfg['resize_method']
        resize_mode = self.test_cfg['resize_mode']

        ori_h, ori_w = data_sample.ori_merged_shape[:2]
        # print(ds.ori_merged_shape)
        if resize_method == 'pad':
            pred_alpha = pred_alpha[:, :ori_h, :ori_w]
        elif resize_method == 'interp':
            pred_alpha = F.interpolate(
                pred_alpha.unsqueeze(0), size=(ori_h, ori_w), mode=resize_mode)
            pred_alpha = pred_alpha[0]  # 1,H,W

        return pred_alpha

    def postprocess(
        self,
        batch_pred_alpha: torch.Tensor,  # N, 1, H, W, float32
        data_samples: List[EditDataSample],
    ) -> List[EditDataSample]:
        """Post-process alpha predictions.

        This function contains the following steps:
            1. Restore padding or interpolation
            1. Mask alpha prediction with trimap
            1. Clamp alpha prediction to 0-1
            1. Convert alpha prediction to uint8
            1. Pack alpha prediction into EditDataSample

        Currently only batch_size 1 is actually supported.

        Args:
            batch_pred_alpha (torch.Tensor): A batch of predicted alpha
                of shape (N, 1, H, W).
            data_samples (List[EditDataSample]): List of data samples.

        Returns:
            List[EditDataSample]: A list of predictions.
                Each data sample contains a pred_alpha,
                which is a torch.Tensor with dtype=uint8, device=cuda:0
        """

        assert batch_pred_alpha.ndim == 4  # N, 1, H, W, float32
        assert len(batch_pred_alpha) == len(data_samples) == 1

        predictions = []
        for pa, ds in zip(batch_pred_alpha, data_samples):
            pa = self.restore_size(pa, ds)  # 1, H, W
            pa = pa[0]  # H, W

            pa.clamp_(min=0, max=1)
            ori_trimap = ds.ori_trimap
            # trimap = torch.from_numpy(ds.ori_trimap).to(pa.device)
            pa[ori_trimap == 255] = 1
            pa[ori_trimap == 0] = 0

            # pa = (trimap == 255) + (trimap == 128) * pa

            pa *= 255
            pa.round_()
            pa = pa.to(dtype=torch.uint8)
            # pa = pa.cpu().numpy()
            pa_sample = EditDataSample(pred_alpha=PixelData(data=pa))
            predictions.append(pa_sample)

        return predictions

    def forward(self,
                batch_inputs: torch.Tensor,
                data_samples: DataSamples = None,
                mode: str = 'feat',
                **kwargs) -> List[EditDataSample]:
        """General forward function.

        Args:
            batch_inputs (torch.Tensor): A batch of inputs.
                with image and trimap concatenated alone channel dimension.
            data_samples (List[EditDataSample], optional):
                A list of data samples, containing:
                - Ground-truth alpha / foreground / background to compute loss
                - other meta information
            mode (str): mode should be one of ``loss``, ``predict`` and
                ``tensor``

                - ``loss``: Called by ``train_step`` and return loss ``dict``
                  used for logging
                - ``predict``: Called by ``val_step`` and ``test_step``
                  and return list of ``BaseDataElement`` results used for
                  computing metric.
                - ``tensor``: Called by custom use to get ``Tensor`` type
                  results.

        Returns:
            List[EditDataElement]:
                Sequence of predictions packed into EditDataElement
        """
        if mode == 'tensor':
            raw = self._forward(batch_inputs, **kwargs)
            return raw
        elif mode == 'predict':
            # Pre-process runs in runner
            batch_inputs = self.resize_inputs(batch_inputs)
            batch_pred_alpha = self._forward_test(batch_inputs, **kwargs)
            predictions = self.postprocess(batch_pred_alpha, data_samples)
            return predictions
        elif mode == 'loss':
            loss = self._forward_train(batch_inputs, data_samples)
            return loss
