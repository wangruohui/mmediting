# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple
import torch

from mmedit.registry import MODELS
from .base_mattor import BaseMattor
from .utils import get_unknown_tensor
from mmedit.data_element import EditDataSample, PixelData

from mmengine.logging import MMLogger


@MODELS.register_module()
class DIM(BaseMattor):
    """Deep Image Matting model.

    https://arxiv.org/abs/1703.03872

    .. note::

        For ``(self.train_cfg.train_backbone, self.train_cfg.train_refiner)``:

            * ``(True, False)`` corresponds to the encoder-decoder stage in \
                the paper.
            * ``(False, True)`` corresponds to the refinement stage in the \
                paper.
            * ``(True, True)`` corresponds to the fine-tune stage in the paper.

    Args:
        backbone (dict): Config of backbone.
        refiner (dict): Config of refiner.
        train_cfg (dict): Config of training. In ``train_cfg``,
            ``train_backbone`` should be specified. If the model has a refiner,
            ``train_refiner`` should be specified.
        test_cfg (dict): Config of testing. In ``test_cfg``, If the model has a
            refiner, ``train_refiner`` should be specified.
        pretrained (str): Path of pretrained model.
        loss_alpha (dict): Config of the alpha prediction loss. Default: None.
        loss_comp (dict): Config of the composition loss. Default: None.
        loss_refine (dict): Config of the loss of the refiner. Default: None.
    """

    def __init__(self,
                 data_preprocessor,
                 backbone,
                 refiner=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_alpha=None,
                 loss_comp=None,
                 loss_refine=None):
        # Build data _preprocessor and backbone
        super().__init__(
            backbone=backbone,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            train_cfg=train_cfg,
            test_cfg=test_cfg)

        # build refiner if it's not None.
        if refiner is None:
            self.train_cfg['train_refiner'] = False
            self.test_cfg['refine'] = False
        else:
            self.refiner = MODELS.build(refiner)

        # if argument train_cfg is not None, validate if the config is proper.
        assert hasattr(self.train_cfg, 'train_refiner')
        assert hasattr(self.test_cfg, 'refine')
        if self.test_cfg.refine and not self.train_cfg.train_refiner:
            logger = MMLogger.create_instance('mmedit')
            logger.warning(
                'You are not training the refiner, but it is used for '
                'model forwarding.')

        if not self.train_cfg.train_backbone:
            self.freeze_backbone()

        if all(v is None for v in (loss_alpha, loss_comp, loss_refine)):
            raise ValueError('Please specify at least one loss for DIM.')

        if loss_alpha is not None:
            self.loss_alpha = MODELS.build(loss_alpha)
        if loss_comp is not None:
            self.loss_comp = MODELS.build(loss_comp)
        if loss_refine is not None:
            self.loss_refine = MODELS.build(loss_refine)

        self.init_weights()

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

    def _forward(self, x: torch.Tensor,
                 refine: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Raw forward function.

        Args:
            x (torch.Tensor): Concatenation of merged image and trimap
                with shape (N, 4, H, W)
            refine (bool): if forward through refiner

        Returns:
            torch.Tensor: pred_alpha, with shape (N, 1, H, W)
            torch.Tensor: pred_refine, with shape (N, 4, H, W)
        """

        raw_alpha = self.backbone(x)
        pred_alpha = raw_alpha.sigmoid()

        if refine:
            refine_input = torch.cat((x[:, :3, :, :], pred_alpha), 1)
            pred_refine = self.refiner(refine_input, raw_alpha)
        else:
            # As ONNX does not support NoneType for output,
            # we choose to use zero tensor to represent None
            pred_refine = torch.zeros([])
        return pred_alpha, pred_refine

    def _forward_test(self, inputs, data_samples):
        """Forward to get alpha prediction."""
        pred_alpha, pred_refine = self._forward(inputs, self.test_cfg.refine)
        if self.test_cfg.refine:
            return pred_refine
        else:
            return pred_alpha

    def _forward_train(self, inputs, data_samples):
        """Defines the computation performed at every training call.

        Args:
            inputs (torch.Tensor): Concatenation of normalized image and trimap
                shape (N, 4, H, W)
            data_samples (list[EditDataSample]): Data samples containing:
                - alpha (Tensor): of shape (N, 1, H, W). Tensor of alpha read by
                    opencv.
                - ori_merged (Tensor): of shape (N, C, H, W). Tensor of origin merged
                    image read by opencv (not normalized).
                - fg (Tensor): of shape (N, C, H, W). Tensor of fg read by opencv.
                - bg (Tensor): of shape (N, C, H, W). Tensor of bg read by opencv.

        Returns:
            dict: Contains the loss items and batch information.
        """
        trimap = inputs[:, 3:, :, :]
        gt_alpha = torch.stack(tuple(ds.gt_alpha.data for ds in data_samples))
        gt_fg = torch.stack(tuple(ds.gt_fg.data for ds in data_samples))
        gt_bg = torch.stack(tuple(ds.gt_bg.data for ds in data_samples))
        gt_merged = torch.stack(
            tuple(ds.gt_merged.data for ds in data_samples))

        # merged, trimap, meta, alpha, ori_merged, fg, bg
        pred_alpha, pred_refine = self._forward(inputs,
                                                self.train_cfg.train_refiner)

        weight = get_unknown_tensor(trimap, unknown_value=128 / 255)

        losses = dict()
        if self.train_cfg.train_backbone:
            if self.loss_alpha is not None:
                losses['loss_alpha'] = self.loss_alpha(pred_alpha, gt_alpha,
                                                       weight)
            if self.loss_comp is not None:
                losses['loss_comp'] = self.loss_comp(pred_alpha, gt_fg, gt_bg,
                                                     gt_merged, weight)
        if self.train_cfg.train_refiner:
            losses['loss_refine'] = self.loss_refine(pred_refine, gt_alpha,
                                                     weight)
        return losses
        return {'losses': losses, 'num_samples': inputs.size(0)}
