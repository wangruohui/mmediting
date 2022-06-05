# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import auto_fp16

from mmedit.registry import MODELS
from .base_mattor import BaseMattor
from .utils import get_unknown_tensor


@MODELS.register_module()
class IndexNet(BaseMattor):
    """IndexNet matting model.

    This implementation follows:
    Indices Matter: Learning to Index for Deep Image Matting

    Args:
        backbone (dict): Config of backbone.
        train_cfg (dict): Config of training. In 'train_cfg', 'train_backbone'
            should be specified.
        test_cfg (dict): Config of testing.
        pretrained (str): path of pretrained model.
        loss_alpha (dict): Config of the alpha prediction loss. Default: None.
        loss_comp (dict): Config of the composition loss. Default: None.
    """

    def __init__(
        self,
        data_preprocessor,
        backbone,
        loss_alpha=None,
        loss_comp=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super().__init__(
            backbone=backbone,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            train_cfg=train_cfg,
            test_cfg=test_cfg)

        self.loss_alpha = (
            MODELS.build(loss_alpha) if loss_alpha is not None else None)
        self.loss_comp = (
            MODELS.build(loss_comp) if loss_comp is not None else None)

    #     # support fp16
    #     self.fp16_enabled = False

    # def forward_dummy(self, inputs):
    #     return self.backbone(inputs)

    @auto_fp16(apply_to=('merged', 'trimap'))
    def forward_train(self, merged, trimap, meta, alpha, ori_merged, fg, bg):
        """Forward function for training IndexNet model.

        Args:
            merged (Tensor): Input images tensor with shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            trimap (Tensor): Tensor of trimap with shape (N, 1, H, W).
            meta (list[dict]): Meta data about the current data batch.
            alpha (Tensor): Tensor of alpha with shape (N, 1, H, W).
            ori_merged (Tensor): Tensor of origin merged images (not
                normalized) with shape (N, C, H, W).
            fg (Tensor): Tensor of foreground with shape (N, C, H, W).
            bg (Tensor): Tensor of background with shape (N, C, H, W).

        Returns:
            dict: Contains the loss items and batch information.
        """
        pred_alpha = self.backbone(torch.cat((merged, trimap), 1))

        losses = dict()
        weight = get_unknown_tensor(trimap, meta)
        if self.loss_alpha is not None:
            losses['loss_alpha'] = self.loss_alpha(pred_alpha, alpha, weight)
        if self.loss_comp is not None:
            losses['loss_comp'] = self.loss_comp(pred_alpha, fg, bg,
                                                 ori_merged, weight)
        return {'losses': losses, 'num_samples': merged.size(0)}

    def _forward_test(self, x):
        pred_alpha = self.backbone(x)
        return pred_alpha
