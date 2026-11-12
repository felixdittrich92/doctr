# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import numpy as np

__all__ = ["trim_margins"]


def trim_margins(img: np.ndarray, margin: int = 2, min_contrast: float = 0.1) -> np.ndarray:
    """Crop the uniform margins around the content of a formula image (e.g. the white space of a layout region)

    The content is made of the pixels darker than the mid-gray level between the darkest and the lightest pixel of
    the image. A uniform image is returned unchanged.

    >>> import numpy as np
    >>> from doctr.models.formula._utils import trim_margins
    >>> img = np.full((64, 64, 3), 255, dtype=np.uint8)
    >>> img[20:30, 10:50] = 0
    >>> trim_margins(img).shape
    (14, 44, 3)

    Args:
        img: the image, of shape (H, W, C) or (H, W), in uint8 or in float in [0, 1]
        margin: number of pixels of margin kept around the content
        min_contrast: minimum contrast (relative to the dynamic range of the dtype) for the image to be trimmed

    Returns:
        the trimmed image (a view of the input)
    """
    gray = img.mean(axis=-1) if img.ndim == 3 else img
    if gray.size == 0:
        return img
    scale = 255.0 if np.issubdtype(img.dtype, np.integer) else 1.0
    low, high = float(gray.min()), float(gray.max())
    if high - low < min_contrast * scale:
        return img
    rows, cols = np.nonzero(gray < (low + high) / 2)
    y0, y1 = max(int(rows.min()) - margin, 0), min(int(rows.max()) + margin + 1, img.shape[0])
    x0, x1 = max(int(cols.min()) - margin, 0), min(int(cols.max()) + margin + 1, img.shape[1])
    return img[y0:y1, x0:x1]
