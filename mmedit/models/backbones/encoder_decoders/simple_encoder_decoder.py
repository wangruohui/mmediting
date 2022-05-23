# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

# from mmedit.models.builder import build_component
from ....registry import MODELS


@MODELS.register_module()
class SimpleEncoderDecoder(nn.Module):
    """Simple encoder-decoder model from matting.

    Args:
        encoder (dict): Config of the encoder.
        decoder (dict): Config of the decoder.
    """

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = MODELS.build(encoder)
        if hasattr(self.encoder, 'out_channels'):
            decoder['in_channels'] = self.encoder.out_channels
        self.decoder = MODELS.build(decoder)

    def init_weights(self, pretrained=None):
        self.encoder.init_weights(pretrained)
        self.decoder.init_weights()

    def forward(self, *args, **kwargs):
        """Forward function.

        Returns:
            Tensor: The output tensor of the decoder.
        """
        out = self.encoder(*args, **kwargs)
        out = self.decoder(out)
        return out
