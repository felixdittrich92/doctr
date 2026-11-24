# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from collections import OrderedDict
from copy import deepcopy
from typing import Any

from torch import nn

from doctr.datasets import VOCABS

from ...modules.layers import RepConvLayer
from ...utils import load_pretrained_params

__all__ = [
    "LiteNet",
    "litenet_tiny",
    "litenet_small",
    "litenet_crop_orientation",
    "litenet_page_orientation",
]

default_cfgs: dict[str, dict[str, Any]] = {
    "litenet_tiny": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": None,
    },
    "litenet_small": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": None,
    },
    "litenet_crop_orientation": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 256, 256),
        "classes": [0, -90, 180, 90],
        "url": None,
    },
    "litenet_page_orientation": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 512, 512),
        "classes": [0, -90, 180, 90],
        "url": None,
    },
}

# (stem channels, channels of the 4 stages, number of blocks of the 4 stages)
_LITENET_SPECS: dict[str, tuple[int, tuple[int, int, int, int], tuple[int, int, int, int]]] = {
    "nano": (32, (32, 64, 96, 128), (1, 1, 2, 1)),
    "tiny": (16, (24, 48, 96, 192), (1, 2, 3, 2)),
    "small": (24, (32, 64, 128, 256), (1, 2, 4, 2)),
}


class LiteBlock(nn.Sequential):
    """Depthwise-separable block of LiteNet: a reparameterizable depthwise KxK convolution (spatial mixing) followed by
    a reparameterizable pointwise convolution (channel mixing), each one followed by a ReLU.

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        stride: stride of the depthwise convolution
        kernel_size: kernel size of the depthwise convolution
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int | tuple[int, int] = 1, kernel_size: int = 3
    ) -> None:
        super().__init__(
            RepConvLayer(in_channels, in_channels, kernel_size, stride=stride, groups=in_channels),
            RepConvLayer(in_channels, out_channels, 1),
        )


class LiteNet(nn.Sequential):
    """LiteNet: an ultra-lightweight convolutional backbone built for fast CPU inference.

    The network is a stem followed by 4 stages of depthwise-separable `LiteBlock`, at strides 2 (stem), 4, 8, 16
    and 32 (stages), or `stem_patch_size` (patch stem) and 2, 4, 8 and 16 times more (stages). Every convolution of
    the stages is a `RepConvLayer` (multi-branch during training, a single convolution after
    `doctr.models.reparameterize`) and the only non-linearity is the ReLU: no squeeze-excitation nor hard-swish, which
    are slow on CPUs and degrade the 8-bit quantization. The children are named "0" (stem) to "4" (last stage), and
    "5" (classification head) if `include_top`, so that the feature maps of the stages can be extracted with
    `IntermediateLayerGetter` by the detection, layout, table and formula models.

    Args:
        stem_channels: number of output channels of the stem
        stage_channels: number of output channels of each stage
        depths: number of blocks of each stage
        rect_strides: if True, the stages 2 to 4 only downsample the height (stride (2, 1)), as needed for text
            recognition: a 32 x W input becomes a 1 x W / 4 feature map
        stem_patch_size: if set, the stem is a non-overlapping P x P convolution of stride P (a "patchify" stem)
            instead of a 3 x 3 convolution of stride 2: the image is downsampled at once, which avoids the costly
            full-resolution layers when the task does not need fine details (e.g. the orientation of word crops)
        include_top: whether to include the classification head
        num_classes: number of output classes
        cfg: the configuration dict of the model
    """

    def __init__(
        self,
        stem_channels: int,
        stage_channels: tuple[int, int, int, int],
        depths: tuple[int, int, int, int],
        rect_strides: bool = False,
        stem_patch_size: int | None = None,
        include_top: bool = True,
        num_classes: int = 1000,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        if stem_patch_size is None:
            layers["0"] = RepConvLayer(3, stem_channels, 3, stride=2)
        else:
            # The batch normalization is fused into the convolution by `doctr.models.reparameterize`
            layers["0"] = nn.Sequential(
                nn.Conv2d(3, stem_channels, stem_patch_size, stride=stem_patch_size, bias=False),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True),
            )
        in_channels = stem_channels
        for idx, (out_channels, depth) in enumerate(zip(stage_channels, depths)):
            stride: int | tuple[int, int] = (2, 1) if rect_strides and idx > 0 else 2
            blocks = [LiteBlock(in_channels, out_channels, stride)]
            blocks += [LiteBlock(out_channels, out_channels) for _ in range(depth - 1)]
            layers[str(idx + 1)] = nn.Sequential(*blocks)
            in_channels = out_channels
        if include_top:
            layers["5"] = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(in_channels, num_classes))
        super().__init__(layers)

        self.channels = [stem_channels, *stage_channels]
        self.cfg = cfg

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def from_pretrained(self, path_or_url: str, **kwargs: Any) -> None:
        """Load pretrained parameters onto the model

        Args:
            path_or_url: the path or URL to the model parameters (checkpoint)
            **kwargs: additional arguments to be passed to `doctr.models.utils.load_pretrained_params`
        """
        load_pretrained_params(self, path_or_url, **kwargs)


def _litenet(
    arch: str,
    size: str,
    pretrained: bool,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> LiteNet:
    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["classes"] = kwargs["classes"]
    # The network is fully convolutional: the input shape only sets the preprocessing of the predictors
    _cfg["input_shape"] = kwargs.pop("input_shape", _cfg["input_shape"])
    kwargs.pop("classes")

    stem_channels, stage_channels, depths = _LITENET_SPECS[size]
    model = LiteNet(stem_channels, stage_channels, depths, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if kwargs["num_classes"] != len(default_cfgs[arch]["classes"]) else None
        model.from_pretrained(default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def litenet_tiny(pretrained: bool = False, **kwargs: Any) -> LiteNet:
    """LiteNet-Tiny: an ultra-lightweight reparameterizable backbone (about 0.1M parameters without the head)

    >>> import torch
    >>> from doctr.models import litenet_tiny
    >>> model = litenet_tiny(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 32, 32), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the LiteNet architecture

    Returns:
        A LiteNet model
    """
    return _litenet("litenet_tiny", "tiny", pretrained, ignore_keys=["5.2.weight", "5.2.bias"], **kwargs)


def litenet_small(pretrained: bool = False, **kwargs: Any) -> LiteNet:
    """LiteNet-Small: an ultra-lightweight reparameterizable backbone (about 0.2M parameters without the head)

    >>> import torch
    >>> from doctr.models import litenet_small
    >>> model = litenet_small(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 32, 32), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the LiteNet architecture

    Returns:
        A LiteNet model
    """
    return _litenet("litenet_small", "small", pretrained, ignore_keys=["5.2.weight", "5.2.bias"], **kwargs)


def litenet_crop_orientation(pretrained: bool = False, **kwargs: Any) -> LiteNet:
    """LiteNet-Nano classifying the orientation of word crops (0, -90, 180, 90 degrees), on 256 x 256 inputs: an 8 x 8
    patch stem and narrow stages (about 40k parameters), as the orientation of a word does not need fine details

    >>> import torch
    >>> from doctr.models import litenet_crop_orientation
    >>> model = litenet_crop_orientation(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 256, 256), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the LiteNet architecture

    Returns:
        A LiteNet model
    """
    kwargs.setdefault("stem_patch_size", 8)
    return _litenet("litenet_crop_orientation", "nano", pretrained, ignore_keys=["5.2.weight", "5.2.bias"], **kwargs)


def litenet_page_orientation(pretrained: bool = False, **kwargs: Any) -> LiteNet:
    """LiteNet-Tiny classifying the orientation of pages (0, -90, 180, 90 degrees), on 512 x 512 inputs

    >>> import torch
    >>> from doctr.models import litenet_page_orientation
    >>> model = litenet_page_orientation(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the LiteNet architecture

    Returns:
        A LiteNet model
    """
    return _litenet("litenet_page_orientation", "tiny", pretrained, ignore_keys=["5.2.weight", "5.2.bias"], **kwargs)
