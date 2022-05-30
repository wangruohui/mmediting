# Copyright (c) OpenMMLab. All rights reserved.
import logging
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import ConfigDict
from mmcv.utils import print_log
from mmengine import BaseModel

from mmedit.data_element import EditDataSample, PixelData
# from mmedit.core.evaluation import connectivity, gradient_error, mse, sad
# from ..base import BaseModel
# from ..builder import build_backbone, build_component
from mmedit.registry import MODELS


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

    def __init__(self,
                 backbone,
                 refiner=None,
                 preprocess_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(preprocess_cfg=preprocess_cfg)

        self.train_cfg = train_cfg if train_cfg is not None else ConfigDict()
        self.test_cfg = test_cfg if test_cfg is not None else ConfigDict()
        self.preprocess_cfg = preprocess_cfg if preprocess_cfg else ConfigDict(
        )

        self.backbone = MODELS.build(backbone)
        # build refiner if it's not None.
        if refiner is None:
            self.train_cfg['train_refiner'] = False
            self.test_cfg['refine'] = False
        else:
            self.refiner = MODELS.build(refiner)

        # if argument train_cfg is not None, validate if the config is proper.
        if train_cfg is not None:
            assert hasattr(self.train_cfg, 'train_refiner')
            assert hasattr(self.test_cfg, 'refine')
            if self.test_cfg.refine and not self.train_cfg.train_refiner:
                print_log(
                    'You are not training the refiner, but it is used for '
                    'model forwarding.', 'root', logging.WARNING)

            if not self.train_cfg.train_backbone:
                self.freeze_backbone()

        # validate if test config is proper
        # if not hasattr(self.test_cfg, 'metrics'):
        #     raise KeyError('Missing key "metrics" in test_cfg')

        # if mmcv.is_list_of(self.test_cfg.metrics, str):
        #     for metric in self.test_cfg.metrics:
        #         if metric not in self.allowed_metrics:
        #             raise KeyError(f'metric {metric} is not supported')
        # elif self.test_cfg.metrics is not None:
        #     raise TypeError('metrics must be None or a list of str')

        self.init_weights(pretrained)

    @property
    def with_refiner(self):
        """Whether the matting model has a refiner.
        """
        return hasattr(self, 'refiner') and self.refiner is not None

    def freeze_backbone(self):
        """Freeze the backbone and only train the refiner.
        """
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the model network weights.

        Args:
            pretrained (str, optional): Path to the pretrained weight.
                Defaults to None.
        """
        if pretrained is not None:
            print_log(f'load model from: {pretrained}', logger='root')
        self.backbone.init_weights(pretrained)
        if self.with_refiner:
            self.refiner.init_weights()

    def preprocess(self, data: List[dict], training: bool
                   ) -> Tuple[torch.Tensor, List[EditDataSample]]:

        batch_inputs, batch_data_samples = super().preprocess(data, training)

        # Collect all trimap as a batch
        batch_trimap = []
        for ds in batch_data_samples:
            batch_trimap.append(ds.trimap.data)
            del ds.trimap
        batch_trimap = torch.stack(batch_trimap)

        proc_trimap = self.preprocess_cfg.get('trimap', 'norm')
        if proc_trimap == 'norm':
            batch_trimap = batch_trimap / 255.0  # uint8->float32
        elif proc_trimap == 'label':
            batch_trimap[batch_trimap == 128] = 1
            batch_trimap[batch_trimap == 255] = 2
            batch_trimap = batch_trimap.to(torch.float32)
        elif proc_trimap == 'onehot':
            batch_trimap[batch_trimap == 128] = 1
            batch_trimap[batch_trimap == 255] = 2
            batch_trimap = F.one_hot(
                batch_trimap.to(torch.long), num_classes=3)
            # N 1 H W C -> N C H W
            batch_trimap = batch_trimap[:, 0, :, :, :]
            batch_trimap = batch_trimap.permute(0, 3, 1, 2)
            batch_trimap = batch_trimap.to(torch.float32)

        assert batch_inputs.ndim == batch_trimap.ndim == 4
        # print(batch_inputs.shape)
        # print(batch_trimap.shape)
        assert batch_inputs.shape[-2:] == batch_trimap.shape[-2:], f"""
            Expect batch_merged.shape[-2:] == batch_trimap.shape[-2:],
            but got {batch_inputs.shape[-2:]} vs {batch_trimap.shape[-2:]}
            """

        # Stack image and trimap along channel dimension
        # Currently, all model do this concat at the start of forwarding
        # and data_sample is a very complex data structure
        # so this is a simple work-around to make codes simpler
        # print(f"batch_trimap.dtype = {batch_trimap.dtype}")

        if not training:
            # Pad the images to align with network downsample factor for testing
            if 'pad_multiple' in self.test_cfg:
                ds_factor = self.test_cfg['pad_multiple']
                batch_inputs, pad_width = self._pad(batch_inputs, ds_factor)
                batch_trimap, pad_width = self._pad(batch_trimap, ds_factor)
                # for ds in batch_data_samples:
                #     ds.set_field(
                #         name='pad_width',
                #         value=pad_width,
                #         field_type='metainfo')
            elif 'resize_to_multiple' in self.test_cfg:
                ds_factor = self.test_cfg['resize_to_multiple']
                batch_inputs, batch_trimap = self._interpolate(
                    batch_inputs, batch_trimap, ds_factor)
            else:
                raise NotImplementedError

        batch_concat = torch.cat((batch_inputs, batch_trimap), dim=1)
        # print(batch_concat.shape)
        # print(batch_concat)

        # print(f"batch_concat.size() = {batch_concat.size()}")
        return batch_concat, batch_data_samples

    def _pad(self, data, ds_factor):
        """Pad the images to align with network downsample factor for testing."""

        # ds_factor should be a property of a given model
        # ds_factor = getattr(self, 'ds_factor', self.test_cfg['pad_multiple'])
        pad_mode = self.test_cfg['pad_mode']

        h, w = data.shape[-2:]  # NCHW

        new_h = ds_factor * ((h - 1) // ds_factor + 1)
        new_w = ds_factor * ((w - 1) // ds_factor + 1)

        pad_h = new_h - h
        pad_w = new_w - w
        pad = (pad_h, pad_w)
        if new_h != h or new_w != w:
            pad_width = (0, pad_w, 0, pad_h)  # torch.pad in reverse order
            data = F.pad(data, pad_width, pad_mode)

        return data, pad

    def _interpolate(self, batch_image, batch_trimap, ds_factor):
        """Interpolate images and trimaps to align the downsample factor."""

        # ds_factor should be a property of a given model
        interp_mode = self.test_cfg['interp_mode']

        assert batch_image.shape[-2:] == batch_trimap.shape[-2:]

        h, w = batch_image.shape[-2:]  # NCHW

        new_h = h - (h % ds_factor)
        new_w = w - (w % ds_factor)

        size = (new_h, new_w)
        if new_h != h or new_w != w:
            batch_image = F.interpolate(
                batch_image, size=size, mode=interp_mode)
            batch_trimap = F.interpolate(
                batch_trimap, size=size, mode='nearest')
            # print(size, h, w)
            # batch_image = F.interpolate(
            #     batch_image, size_factor=ds_factor, mode=interp_mode)
            # batch_trimap = F.interpolate(
            #     batch_trimap, size_factor=ds_factor, mode=interp_mode)

        return batch_image, batch_trimap

    def restore_shape_dep(self, pred_alpha, data_sample):
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
        inputs: torch.Tensor,  # N, 4, H, W, float32
        pred_alpha: torch.Tensor,  # N, 1, H, W, float32
        data_samples: List[EditDataSample],
    ) -> List[EditDataSample]:
        """Post-processing for alpha predictions.

        1. Restore padded shape
        1. Mask with trimap
        1. clamp to 0-1
        1. to uint8
        """

        assert pred_alpha.ndim == 4  # N, 1, H, W, float32
        # pred_alpha = pred_alpha[:, 0, :, :]

        # trimap = inputs[:, -1, :, :]

        # pred_alpha.clamp_(min=0, max=1)
        # pred_alpha[trimap == 1] = 1
        # pred_alpha[trimap == 0] = 0
        # pred_alpha *= 255
        # pred_alpha.round_()
        # pred_alpha = pred_alpha.to(dtype=torch.uint8)

        predictions = []
        for pa, ds in zip(pred_alpha, data_samples):
            # pa = self.restore_shape(pa, ds)
            ori_h, ori_w = ds.ori_merged_shape[:2]
            # print(ds.ori_merged_shape)
            if 'pad_multiple' in self.test_cfg:
                pa = pa[:, :ori_h, :ori_w]
            elif 'resize_to_multiple' in self.test_cfg:
                # print(pa.shape)
                pa = F.interpolate(
                    pa.unsqueeze(0),
                    size=(ori_h, ori_w),
                    mode=self.test_cfg['interp_mode'])[0]
                # print(pa.shape)

            pa = pa[0]
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
        # eval_result = self.evaluate(pred_alpha, meta)

        #     if save_image:
        #         self.save_image(pred_alpha, meta, save_path, iteration)

        #     return {'pred_alpha': pred_alpha, 'eval_result': eval_result}

    @abstractmethod
    def _forward_test(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward from inputs to pred_alpha in test mode."""

    def forward(self,
                inputs: torch.Tensor,
                data_samples: List[EditDataSample],
                return_loss=False) -> List[EditDataSample]:
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
        # assert len(inputs) == 1, (
        #     'Currently only batch size=1 is supported, '
        #     'because different image can be of different resolution')

        # import time, torch
        # start = time.time()
        # torch.cuda.synchronize()
        pred_alpha = self._forward_test(inputs)
        predictions = self.postprocess(inputs, pred_alpha, data_samples)
        # torch.cuda.synchronize()
        # middle = time.time()
        return predictions
