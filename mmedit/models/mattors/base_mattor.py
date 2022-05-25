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

    # allowed_metrics = {
    #     'SAD': sad,
    #     'MSE': mse,
    #     'GRAD': gradient_error,
    #     'CONN': connectivity
    # }

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

        batch_trimap = []
        for ds in batch_data_samples:
            batch_trimap.append(ds.trimap.data)
            del ds.trimap
        batch_trimap = torch.stack(batch_trimap)

        assert batch_inputs.ndim == batch_trimap.ndim == 4
        assert batch_inputs.shape[-2:] == batch_trimap.shape[-2:], f"""
            Expect batch_merged.shape[-2:] == batch_trimap.shape[-2:],
            but got {batch_inputs.shape[-2:]} vs {batch_trimap.shape[-2:]}
            """

        # Currently, all model do this concat at the start of forwarding
        # and data_sample is a very complex data structure
        # so this is a simple work-around to make codes simpler
        batch_concat = torch.cat((batch_inputs, batch_trimap / 255.), dim=1)

        if not training:
            # Pad the images to align with network downsample factor for testing
            batch_concat, pad_width = self.pad(batch_concat)

            for ds in batch_data_samples:
                ds.set_field(
                    name='pad_width', value=pad_width, field_type='metainfo')
        #         print(ds)

        # print(batch_concat.shape)
        # print(batch_concat)

        return batch_concat, batch_data_samples

    def pad(self, data):
        """Pad the images to align with network downsample factor for testing."""

        # ds_factor should be a property of a given model
        ds_factor = getattr(self, 'ds_factor', self.test_cfg['pad_multiple'])
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
        pred_alpha = np.clip(pred_alpha, 0, 1)
        pred_alpha[ori_trimap == 0] = 0.
        pred_alpha[ori_trimap == 255] = 1.

        pred_alpha = np.round(pred_alpha * 255).astype(np.uint8)
        # print(pred_alpha.sum())
        return pred_alpha

    @abstractmethod
    def _forward(self, inputs: torch.Tensor, refine: bool) -> torch.Tensor:
        """Forward from cat(image, trimap) to pred_alpha.

        This is the core forward function of the model,
        it is used for both train and test.
        Submodules should rewrite this function.
        """

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
        assert len(inputs) == 1, (
            'Currently only batch size=1 is supported, '
            'because different image can be of different resolution')

        # print(inputs)
        # print(inputs.min())
        # print(inputs.max())
        # print(inputs.sum())
        pred_alpha, pred_refine = self._forward(inputs, self.test_cfg.refine)
        if self.test_cfg.refine:
            pred_alpha = pred_refine

        pred_alpha = pred_alpha.detach().cpu().numpy()
        assert pred_alpha.ndim == 4  # 1, 1, H, W, float32

        predictions = []
        for pa, ds in zip(pred_alpha, data_samples):
            pa = self.restore_shape(pa, ds)
            pa_sample = EditDataSample(pred_alpha=pa)
            # No PixelData as it will shift 2-dim to 3-dim
            predictions.append(pa_sample)

        return predictions
        # eval_result = self.evaluate(pred_alpha, meta)

        #     if save_image:
        #         self.save_image(pred_alpha, meta, save_path, iteration)

        #     return {'pred_alpha': pred_alpha, 'eval_result': eval_result}

    def forward_train(self,
                      inputs: torch.Tensor,
                      data_samples: List[EditDataSample],
                      return_loss=False):
        """_summary_

        Args:
            inputs (torch.Tensor): _description_
            data_samples (List[InstanceData]): _description_
            return_loss (bool, optional): _description_. Defaults to False.
        """

        # return self.forward_train(merged, trimap, meta, alpha, **kwargs)
