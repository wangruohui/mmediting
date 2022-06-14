# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from typing import Dict, List, Optional, Tuple, Union

# import mmcv
import torch
import torch.nn.functional as F
from mmcv import ConfigDict
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.model import BaseModel

from mmedit.data_element import EditDataSample, PixelData
# from mmedit.core.evaluation import connectivity, gradient_error, mse, sad
# from ..base import BaseModel
# from ..builder import build_backbone, build_component
from mmedit.registry import MODELS

# from mmengine.utils import stack_batch
# 2022-02-20 22:26:38,860 - mmdet - INFO - this is a test

DataSamples = Optional[Union[list, torch.Tensor]]
ForwardResults = Union[Dict[str, torch.Tensor], List[EditDataSample],
                       Tuple[torch.Tensor], torch.Tensor]


def _pad(batch_image, ds_factor, mode='reflect'):
    """Pad image to a multiple of give downsampling factor."""

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
    """Resize image to multiple of give downsampling factor."""

    h, w = batch_image.shape[-2:]  # NCHW

    new_h = h - (h % ds_factor)
    new_w = w - (w % ds_factor)

    size = (new_h, new_w)
    if new_h != h or new_w != w:
        batch_image = F.interpolate(batch_image, size=size, mode=mode)

    return batch_image, size


# class TrimapBasedMattor(BaseModel, metaclass=ABCMeta):
class BaseMattor(BaseModel, metaclass=ABCMeta):
    """Base class for matting models.

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
            logger.warn(f'Loading model from: {pretrained} ...')
            self.backbone.init_weights(pretrained)
        # if self.with_refiner:
        #     self.refiner.init_weights()

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
            pred_alpha (np.ndarray shape=(1,H,W)): a single predicted alpha.
            meta (list[dict]): Meta data about the current data batch.
                Currently only batch_size 1 is supported.

        Returns:
            np.ndarray: The reshaped predicted alpha.
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

    """
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
    """

    def postprocess(
        self,
        batch_pred_alpha: torch.Tensor,  # N, 1, H, W, float32
        data_samples: List[EditDataSample],
    ) -> List[EditDataSample]:
        """Post-processing for alpha predictions.

        1. Restore padded shape
        1. Mask with trimap
        1. clamp to 0-1
        1. to uint8
        """

        assert batch_pred_alpha.ndim == 4  # N, 1, H, W, float32
        assert len(batch_pred_alpha) == len(data_samples) == 1
        # pred_alpha = pred_alpha[:, 0, :, :]

        # trimap = inputs[:, -1, :, :]

        # pred_alpha.clamp_(min=0, max=1)
        # pred_alpha[trimap == 1] = 1
        # pred_alpha[trimap == 0] = 0
        # pred_alpha *= 255
        # pred_alpha.round_()
        # pred_alpha = pred_alpha.to(dtype=torch.uint8)

        predictions = []
        for pa, ds in zip(batch_pred_alpha, data_samples):
            pa = self.restore_size(pa, ds)  # 1,H,W
            pa = pa[0]  # H,W

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
            # No PixelData as it will shift 2-dim to 3-dim
            predictions.append(pa_sample)
        # end = time.time()
        # torch.cuda.synchronize()

        # print("time: ", end - middle, middle - start)
        return predictions

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
            batch_inputs = self.resize_inputs(batch_inputs)
            batch_pred_alpha = self._forward_test(batch_inputs, data_samples)
            predictions = self.postprocess(batch_pred_alpha, data_samples)
            return predictions
        elif mode == 'loss':
            loss = self._forward_train(batch_inputs, data_samples)
            return loss

    def forward_dummy(self, inputs):
        """For ONNX."""
        return self._forward(inputs, self.with_refiner)
