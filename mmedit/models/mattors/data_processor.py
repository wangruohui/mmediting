# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.model import BaseDataPreprocessor

from mmedit.data_element import EditDataSample, PixelData
from mmedit.registry import MODELS

DataSamples = Optional[Union[list, torch.Tensor]]
ForwardResults = Union[Dict[str, torch.Tensor], List[EditDataSample],
                       Tuple[torch.Tensor], torch.Tensor]


def _pad(batch_image, ds_factor, mode='reflect'):
    """Pad image size to a multiple of give downsampling factor."""

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
    """Interpolate image size to a multiple of give downsampling factor."""

    h, w = batch_image.shape[-2:]  # NCHW

    new_h = h - (h % ds_factor)
    new_w = w - (w % ds_factor)

    size = (new_h, new_w)
    if new_h != h or new_w != w:
        batch_image = F.interpolate(batch_image, size=size, mode=mode)

    return batch_image, size


@MODELS.register_module()
class ImageAndTrimapPreprocessor(BaseDataPreprocessor):
    """Image and Trimap pre-processor for trimap-based matting models.

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
        to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        device (int or torch.device): Target device.
    """

    def __init__(self,
                 mean: float = [123.675, 116.28, 103.53],
                 std: float = [58.395, 57.12, 57.375],
                 to_rgb: bool = True,
                 inputs_only=False,
                 trimap_proc: str = 'rescale_to_zero_one',
                 size_divisor: int = 32,
                 resize_method: str = 'pad',
                 resize_mode: str = 'reflect',
                 device: Union[int, torch.device] = 'cpu'):
        super().__init__(device)
        self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1), False)
        self.register_buffer('std', torch.tensor(std).view(-1, 1, 1), False)
        self.trimap_proc = trimap_proc
        self.size_divisor = size_divisor
        self.resize_method = resize_method
        self.resize_mode = resize_mode
        self.to_rgb = to_rgb
        self.inputs_only = inputs_only

    def _proc_inputs(self, inputs: List[torch.Tensor]):
        # bgr to rgb
        if self.to_rgb and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]
        # Normalization.
        inputs = [(_input - self.mean) / self.std for _input in inputs]
        # Stack Tensor.
        batch_inputs = torch.stack(inputs)
        assert batch_inputs.ndim == 4

        return batch_inputs

    def _proc_images_in_data_sample(self, data_samples, key):
        for ds in data_samples:
            # bgr to rgb
            value = getattr(ds, key)
            ispixeldata = False
            if isinstance(value, PixelData):
                ispixeldata = True
                value = value.data

            if self.to_rgb and value[0].size(0) == 3:
                value = value[[2, 1, 0], ...]
            # Rescale to 0 to 1 for gt_fg/gt_bg/gt_merged
            # No inplace here!
            value = value / 255.0
            # Stack Tensor.
            assert value.ndim == 3

            if ispixeldata:
                value = PixelData(data=value)
            setattr(ds, key, value)

        return data_samples

    def _proc_trimap(self, trimaps: List[torch.Tensor]):
        batch_trimaps = torch.stack(trimaps)

        if self.trimap_proc == 'rescale_to_zero_one':
            batch_trimaps = batch_trimaps / 255.0  # uint8->float32
        elif self.trimap_proc == 'label':
            batch_trimaps[batch_trimaps == 128] = 1
            batch_trimaps[batch_trimaps == 255] = 2
            batch_trimaps = batch_trimaps.to(torch.float32)
        elif self.trimap_proc == 'onehot':
            batch_trimaps[batch_trimaps == 128] = 1
            batch_trimaps[batch_trimaps == 255] = 2
            batch_trimaps = F.one_hot(
                batch_trimaps.to(torch.long), num_classes=3)
            # N 1 H W C -> N C H W
            batch_trimaps = batch_trimaps[:, 0, :, :, :]
            batch_trimaps = batch_trimaps.permute(0, 3, 1, 2)
            batch_trimaps = batch_trimaps.to(torch.float32)

        return batch_trimaps

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
            assert len(data) == 1, ("only batch_size=1 "
                                    "is supported for testing.")

        images, trimaps, batch_data_samples = self.collate_data(data)

        batch_images = self._proc_inputs(images)
        batch_trimaps = self._proc_trimap(trimaps)

        if training and (not self.inputs_only):
            self._proc_images_in_data_sample(batch_data_samples, 'gt_fg')
            self._proc_images_in_data_sample(batch_data_samples, 'gt_bg')
            self._proc_images_in_data_sample(batch_data_samples, 'gt_merged')
            self._proc_images_in_data_sample(batch_data_samples, 'gt_alpha')

        # Stack image and trimap along channel dimension
        # Currently, all model do this concat at the start of forwarding
        # and data_sample is a very complex data structure
        # so this is a simple work-around to make codes simpler
        # print(f"batch_trimap.dtype = {batch_trimap.dtype}")

        if not training:
            # Pad the images to align with network downsample factor for testing
            if self.resize_method == 'pad':
                batch_images, _ = _pad(batch_images, self.size_divisor,
                                       self.resize_mode)
                batch_trimaps, _ = _pad(batch_trimaps, self.size_divisor,
                                        self.resize_mode)
            elif self.resize_method == 'interp':
                batch_images, _ = _interpolate(batch_images, self.size_divisor,
                                               self.resize_mode)
                batch_trimaps, _ = _interpolate(batch_trimaps,
                                                self.size_divisor, 'nearest')
            else:
                raise NotImplementedError

        assert batch_images.ndim == batch_trimaps.ndim == 4
        assert batch_images.shape[-2:] == batch_trimaps.shape[-2:], f"""
            Expect merged.shape[-2:] == trimap.shape[-2:],
            but got {batch_images.shape[-2:]} vs {batch_trimaps.shape[-2:]}
            """

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

    def postprocess(
        self,
        # inputs: torch.Tensor,  # N, 4, H, W, float32
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
        assert len(pred_alpha) == len(data_samples) == 1
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
            if self.resize_method == 'pad':
                pa = pa[:, :ori_h, :ori_w]
            elif self.resize_method == 'interp':
                pa = F.interpolate(
                    pa.unsqueeze(0),
                    size=(ori_h, ori_w),
                    mode=self.resize_mode)
                pa = pa[0]

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
