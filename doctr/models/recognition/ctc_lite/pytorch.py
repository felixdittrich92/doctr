# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any

import torch
from torch import nn

from doctr.datasets import VOCABS

from ..._utils import ConfidenceAggregation
from ...classification import litenet_tiny
from ..crnn.pytorch import CRNN, CTCPostProcessor

__all__ = ["CTCLite", "ctc_lite"]

default_cfgs: dict[str, dict[str, Any]] = {
    "ctc_lite": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["french"],
        "url": None,
    },
}


class SequenceMixerBlock(nn.Module):
    """Convolutional sequence mixing block, replacing a recurrent layer: a depthwise convolution along the sequence
    (context of the neighbouring timesteps) and a pointwise feed-forward network (channel mixing), with a residual
    connection. It works on (N, C, 1, T) maps, so that the inference graph only holds 2D convolutions (the batch
    normalizations are fused into them by `doctr.models.reparameterize`), which are fast and quantization-friendly.

    Args:
        channels: number of channels
        kernel_size: size of the depthwise convolution along the sequence
        expansion: expansion ratio of the feed-forward network
    """

    def __init__(self, channels: int, kernel_size: int = 7, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.mixer = nn.Sequential(
            nn.Conv2d(channels, channels, (1, kernel_size), padding=(0, kernel_size // 2), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.mixer(x))


class CTCLite(CRNN):
    """Ultra-lightweight CTC text recognition model for fast CPU inference: a LiteNet backbone with rectangular strides
    (a 32 x W crop becomes a sequence of W / 4 feature vectors), convolutional sequence mixing blocks instead of the
    bidirectional LSTM of CRNN, and a linear CTC classifier. Same training target, loss and decoding as CRNN.

    The model is fully convolutional along the width: once trained, it reads crops of any width (e.g. text lines) and
    the number of timesteps grows with it.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        num_blocks: number of sequence mixing blocks
        kernel_size: size of the depthwise convolutions along the sequence
        expansion: expansion ratio of the feed-forward networks of the sequence mixing blocks
        input_shape: shape of the input crops
        exportable: onnx exportable returns only logits
        cfg: configuration dictionary
        confidence_aggregation: aggregation method of the character probabilities into the word confidence
    """

    _children_names: list[str] = ["feat_extractor", "decoder", "linear", "postprocessor"]

    def __init__(
        self,
        feature_extractor: nn.Module,
        vocab: str,
        num_blocks: int = 2,
        kernel_size: int = 7,
        expansion: int = 1,
        input_shape: tuple[int, int, int] = (3, 32, 128),
        exportable: bool = False,
        cfg: dict[str, Any] | None = None,
        confidence_aggregation: ConfidenceAggregation = "min",
    ) -> None:
        # Skip the initialization of the recurrent layers of CRNN
        super(CRNN, self).__init__()
        self.vocab = vocab
        self.cfg = cfg
        self.max_length = 32
        self.exportable = exportable
        self.feat_extractor = feature_extractor

        # Resolve the number of channels of the feature map
        _is_training = self.feat_extractor.training
        self.feat_extractor.eval()
        with torch.inference_mode():
            channels = self.feat_extractor(torch.zeros((1, *input_shape))).shape[1]
        self.feat_extractor.train(_is_training)

        self.decoder = nn.Sequential(  # type: ignore[assignment]
            *[SequenceMixerBlock(channels, kernel_size, expansion) for _ in range(num_blocks)]
        )
        self.linear = nn.Linear(in_features=channels, out_features=len(vocab) + 1)

        self.postprocessor = CTCPostProcessor(vocab=vocab, confidence_aggregation=confidence_aggregation)

        for n, m in self.named_modules():
            # Don't override the initialization of the backbone
            if n.startswith("feat_extractor."):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def _compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feat_extractor(x)
        # Collapse the remaining height (1 for 32-pixel high crops): (N, C, 1, T)
        if features.shape[2] != 1:
            features = features.mean(dim=2, keepdim=True)
        features = self.decoder(features)
        # (N, C, 1, T) --> (N, T, C)
        return self.linear(features.squeeze(2).transpose(1, 2))


def _ctc_lite(
    arch: str,
    pretrained: bool,
    pretrained_backbone: bool = True,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> CTCLite:
    pretrained_backbone = pretrained_backbone and not pretrained

    # Feature extractor: LiteNet with rectangular strides, without its classification head
    feat_extractor = litenet_tiny(pretrained=pretrained_backbone, rect_strides=True, include_top=False)

    kwargs["vocab"] = kwargs.get("vocab", default_cfgs[arch]["vocab"])
    kwargs["input_shape"] = kwargs.get("input_shape", default_cfgs[arch]["input_shape"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["vocab"] = kwargs["vocab"]
    _cfg["input_shape"] = kwargs["input_shape"]

    # Build the model
    model = CTCLite(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if _cfg["vocab"] != default_cfgs[arch]["vocab"] else None
        model.from_pretrained(default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def ctc_lite(pretrained: bool = False, **kwargs: Any) -> CTCLite:
    """Ultra-lightweight CTC text recognition model for fast CPU inference (cf. `CTCLite`): a LiteNet-Tiny backbone and
    two convolutional sequence mixing blocks, about 0.27M parameters.

    >>> import torch
    >>> from doctr.models import ctc_lite
    >>> model = ctc_lite(pretrained=False)
    >>> input_tensor = torch.rand(1, 3, 32, 128)
    >>> out = model(input_tensor)

    Args:
        pretrained: If True, returns a model pre-trained on our text recognition dataset
        **kwargs: keyword arguments of the CTCLite architecture

    Returns:
        text recognition architecture
    """
    return _ctc_lite("ctc_lite", pretrained, ignore_keys=["linear.weight", "linear.bias"], **kwargs)
