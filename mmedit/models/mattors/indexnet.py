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

    def __init__(self,
                 data_preprocessor,
                 backbone,
                 loss_alpha=None,
                 loss_comp=None,
                 init_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__(
            backbone=backbone,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
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

    def _forward(self, inputs):
        pred_alpha = self.backbone(inputs)
        return pred_alpha

    def _forward_test(self, inputs):
        return self._forward(inputs)

    # @auto_fp16(apply_to=('merged', 'trimap'))
    def _forward_train(self, inputs, data_samples):
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
        # for k, v in self.named_parameters():
        #     vnan = v.isnan().any()
        #     if vnan:
        #         print(k, vnan)
        #         exit()

        trimap = inputs[:, 3:, :, :]
        gt_alpha = torch.stack(tuple(ds.gt_alpha.data for ds in data_samples))
        gt_fg = torch.stack(tuple(ds.gt_fg.data for ds in data_samples))
        gt_bg = torch.stack(tuple(ds.gt_bg.data for ds in data_samples))
        gt_merged = torch.stack(
            tuple(ds.gt_merged.data for ds in data_samples))

        pred_alpha = self.backbone(inputs)

        # if (torch.isnan(inputs).any() or torch.isnan(trimap).any()
        #         or torch.isnan(inputs[:, :3, :, :]).any()
        #         or torch.isnan(pred_alpha).any()):
        #     print("inputs", torch.isnan(inputs).any())
        #     print("trimap", torch.isnan(trimap).any())
        #     print("merged", torch.isnan(inputs[:, :3, :, :]).any())
        #     print("pred_alpha", torch.isnan(pred_alpha).any())
        #     print(losses)
        #     exit()
        weight = get_unknown_tensor(trimap, unknown_value=128 / 255)

        losses = dict()
        # print("gt_alpha", torch.isnan(gt_alpha).any())
        # print("weight", torch.isnan(weight).any())
        # print("gt_fg", torch.isnan(gt_fg).any())
        # print("gt_bg", torch.isnan(gt_bg).any())
        # print("gt_merged", torch.isnan(gt_merged).any())

        if self.loss_alpha is not None:
            losses['loss_alpha'] = self.loss_alpha(pred_alpha, gt_alpha,
                                                   weight)
        if self.loss_comp is not None:
            losses['loss_comp'] = self.loss_comp(pred_alpha, gt_fg, gt_bg,
                                                 gt_merged, weight)

        return losses
        # return {'losses': losses, 'num_samples': merged.size(0)}
