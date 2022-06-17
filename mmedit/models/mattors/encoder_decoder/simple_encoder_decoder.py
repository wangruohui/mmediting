# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
from mmengine.model import BaseModule

from mmedit.registry import MODELS


@MODELS.register_module()
class SimpleEncoderDecoder(BaseModule):
    """Simple encoder-decoder model from matting.

    Args:
        encoder (dict): Config of the encoder.
        decoder (dict): Config of the decoder.
    """

    def __init__(self,
                 encoder: dict,
                 decoder: dict,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)

        self.encoder = MODELS.build(encoder)
        if hasattr(self.encoder, 'out_channels'):
            decoder['in_channels'] = self.encoder.out_channels
        self.decoder = MODELS.build(decoder)

    # def init_weights(self, pretrained=None):
    #     self.encoder.init_weights(pretrained)
    #     self.decoder.init_weights()

    def forward(self, *args, **kwargs):
        """Forward function.

        Returns:
            Tensor: The output tensor of the decoder.
        """
        out = self.encoder(*args, **kwargs)
        out = self.decoder(out)
        return out
