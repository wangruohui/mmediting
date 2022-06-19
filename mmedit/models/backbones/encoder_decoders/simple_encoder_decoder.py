# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmedit.models.builder import build_component
from mmedit.models.registry import BACKBONES


@BACKBONES.register_module()
class SimpleEncoderDecoder(nn.Module):
    """Simple encoder-decoder model from matting.

    Args:
        encoder (dict): Config of the encoder.
        decoder (dict): Config of the decoder.
    """

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = build_component(encoder)
        if hasattr(self.encoder, 'out_channels'):
            decoder['in_channels'] = self.encoder.out_channels
        self.decoder = build_component(decoder)

    def init_weights(self, pretrained=None):
        self.encoder.init_weights(pretrained)
        self.decoder.init_weights()

    def forward(self, *args, **kwargs):
        """Forward function.

        Returns:
            Tensor: The output tensor of the decoder.
        """
        import thckpt
        exp = thckpt.Experiment(exp_name='master', root_dir='d:\exp')
        # args = exp.checkpoint(args, 'args')
        out = self.encoder(*args, **kwargs)
        # out = exp.checkpoint(out, 'enc-out')
        # out = exp.save(out, 'enc-out-save')
        out = self.decoder(out)
        # out = exp.checkpoint(out, 'dec-out')
        # out = exp.save(out, 'dec-out-save')

        return out
