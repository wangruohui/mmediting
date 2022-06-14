# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
# import torch.nn.functional as F
from mmengine.model import BaseDataPreprocessor

from mmedit.data_element import EditDataSample, PixelData
from mmedit.registry import MODELS

DataSamples = Optional[Union[list, torch.Tensor]]
ForwardResults = Union[Dict[str, torch.Tensor], List[EditDataSample],
                       Tuple[torch.Tensor], torch.Tensor]


@MODELS.register_module()
class MattorPreprocessor(BaseDataPreprocessor):
    """Image and trimap pre-processor for trimap-based matting models.

    Accept the data sampled by the dataLoader, and preprocesses it into the
    format of the model input. ``ImgDataPreprocessor`` provides the
    basic data pre-processing as follows

    - Collate and move data to the target device.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.

    For ``ImgDataPreprocessor``, the dimension of the single inputs must be
    (3, H, W).

    Note:
        ``ImgDataPreprocessor`` and its subclass is built in the
        constructor of :class:`BaseDataset`.

    Args:
        mean (Sequence[float or int]): The pixel mean of R, G, B channels.
            Defaults to (127.5, 127.5, 127.5).
        std (Sequence[float or int]): The pixel standard deviation of R, G, B
            channels. (127.5, 127.5, 127.5)
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
    """

    def __init__(self,
                 mean: float = [123.675, 116.28, 103.53],
                 std: float = [58.395, 57.12, 57.375],
                 bgr_to_rgb: bool = True,
                 proc_inputs: str = 'normalize',
                 proc_trimap: str = 'rescale_to_zero_one',
                 proc_gt: str = 'rescale_to_zero_one'):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1), False)
        self.register_buffer('std', torch.tensor(std).view(-1, 1, 1), False)
        self.bgr_to_rgb = bgr_to_rgb
        self.proc_inputs = proc_inputs
        self.proc_trimap = proc_trimap
        self.proc_gt = proc_gt

    def _proc_inputs(self, inputs: List[torch.Tensor]):
        if self.proc_inputs == 'normalize':
            # bgr to rgb
            if self.bgr_to_rgb and inputs[0].size(0) == 3:
                inputs = [_input[[2, 1, 0], ...] for _input in inputs]
            # Normalization.
            inputs = [(_input - self.mean) / self.std for _input in inputs]
            # Stack Tensor.
            batch_inputs = torch.stack(inputs)
        else:
            raise ValueError(
                f'proc_inputs = {self.proc_inputs} is not supported.')

        assert batch_inputs.ndim == 4
        return batch_inputs

    def _proc_trimap(self, trimaps: List[torch.Tensor]):
        batch_trimaps = torch.stack(trimaps)

        if self.proc_trimap == 'rescale_to_zero_one':
            batch_trimaps = batch_trimaps / 255.0  # uint8->float32
        elif self.proc_trimap == 'as_is':
            batch_trimaps = batch_trimaps.to(torch.float32)
        # elif self.trimap_proc == 'label':
        #     batch_trimaps[batch_trimaps == 128] = 1
        #     batch_trimaps[batch_trimaps == 255] = 2
        #     batch_trimaps = batch_trimaps.to(torch.float32)
        # elif self.trimap_proc == 'onehot':
        #     batch_trimaps[batch_trimaps == 128] = 1
        #     batch_trimaps[batch_trimaps == 255] = 2
        #     batch_trimaps = F.one_hot(
        #         batch_trimaps.to(torch.long), num_classes=3)
        #     # N 1 H W C -> N C H W
        #     batch_trimaps = batch_trimaps[:, 0, :, :, :]
        #     batch_trimaps = batch_trimaps.permute(0, 3, 1, 2)
        #     batch_trimaps = batch_trimaps.to(torch.float32)
        else:
            raise ValueError(
                f'proc_trimap = {self.proc_trimap} is not supported.')

        return batch_trimaps

    def _proc_gt(self, data_samples, key):
        assert key.startswith('gt')
        # Rescale gt_fg / gt_bg / gt_merged / gt_alpha to 0 to 1
        if self.proc_gt == 'rescale_to_zero_one':
            for ds in data_samples:
                try:
                    value = getattr(ds, key)
                except AttributeError:
                    continue

                ispixeldata = isinstance(value, PixelData)
                if ispixeldata:
                    value = value.data

                # !! DO NOT process trimap here, as trimap may have dim == 3
                if self.bgr_to_rgb and value[0].size(0) == 3:
                    value = value[[2, 1, 0], ...]

                value = value / 255.0  # uint8 -> float32 No inplace here

                assert value.ndim == 3

                if ispixeldata:
                    value = PixelData(data=value)
                setattr(ds, key, value)
        elif self.proc_gt == 'ignore':
            pass
        else:
            raise ValueError(f'proc_gt = {self.proc_gt} is not supported.')

        return data_samples

    def forward(self,
                data: Sequence[dict],
                training: bool = False) -> Tuple[torch.Tensor, list]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (Sequence[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Tuple[torch.Tensor, list]: Data in the same format as the model
            input.
        """
        if not training:
            # Image may of different size when testing
            assert len(data) == 1, ('only batch_size=1 '
                                    'is supported for testing.')

        images, trimaps, batch_data_samples = self.collate_data(data)

        batch_images = self._proc_inputs(images)
        batch_trimaps = self._proc_trimap(trimaps)

        if training:
            self._proc_gt(batch_data_samples, 'gt_fg')
            self._proc_gt(batch_data_samples, 'gt_bg')
            self._proc_gt(batch_data_samples, 'gt_merged')
            # For training, gt_alpha ranges from 0-1, is used to compute loss
            # For testing, ori_alpha is used
            self._proc_gt(batch_data_samples, 'gt_alpha')

        # if not training:
        #     # Pad the images to align with network downsample factor for testing  # noqa
        #     if self.resize_method == 'pad':
        #         batch_images, _ = _pad(batch_images, self.size_divisor,
        #                                self.resize_mode)
        #         batch_trimaps, _ = _pad(batch_trimaps, self.size_divisor,
        #                                 self.resize_mode)
        #     elif self.resize_method == 'interp':
        #         batch_images, _ = _interpolate(batch_images, self.size_divisor,  # noqa
        #                                        self.resize_mode)
        #         batch_trimaps, _ = _interpolate(batch_trimaps,
        #                                         self.size_divisor, 'nearest')
        #     else:
        #         raise NotImplementedError

        # Stack image and trimap along channel dimension
        # All existing models do concat at the start of forwarding
        # and data_sample is a very complex data structure
        # so this is a simple work-around to make codes simpler
        # print(f"batch_trimap.dtype = {batch_trimap.dtype}")

        assert batch_images.ndim == batch_trimaps.ndim == 4
        assert batch_images.shape[-2:] == batch_trimaps.shape[-2:], f"""
            Expect merged.shape[-2:] == trimap.shape[-2:],
            but got {batch_images.shape[-2:]} vs {batch_trimaps.shape[-2:]}
            """

        # N, (4/6), H, W
        batch_inputs = torch.cat((batch_images, batch_trimaps), dim=1)

        return batch_inputs, batch_data_samples

    def collate_data(self, data: Sequence[dict]) -> Tuple[list, list, list]:
        """Collating and moving data to the target device.

        Take the data sampled from dataloader and unpack them into list of
        tensor and list of labels. Then moving tensor to the target device.

        Subclass could override it to be compatible with the custom format
        data sampled from custom dataloader.

        Args:
            data (Sequence[dict]): data sampled from dataloader.

        Returns:
            Tuple[List[torch.Tensor], list]: Unstacked list of input tensor
            and list of labels at target device.
        """
        inputs = [data_['inputs'] for data_ in data]
        trimaps = [data_['data_sample'].trimap.data for data_ in data]
        batch_data_samples = [data_['data_sample'] for data_ in data]

        # Move data from CPU to corresponding device.
        inputs = [_input.to(self.device) for _input in inputs]
        trimaps = [_trimap.to(self.device) for _trimap in trimaps]
        batch_data_samples = [
            data_sample.to(self.device) for data_sample in batch_data_samples
        ]
        return inputs, trimaps, batch_data_samples
