# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import List, Optional, Tuple

import numpy as np

from doctr.datasets import encode_sequences
from doctr.utils.repr import NestedObject

__all__ = ["RecognitionPostProcessor", "RecognitionModel"]


class RecognitionModel(NestedObject):
    """Implements abstract RecognitionModel class"""

    vocab: str
    max_length: int

    def build_target(
        self,
        gts: List[str],
    ) -> Tuple[np.ndarray, List[int]]:
        """Encode a list of gts sequences into a np array and gives the corresponding*
        sequence lengths.

        Args:
        ----
            gts: list of ground-truth labels

        Returns:
        -------
            A tuple of 2 tensors: Encoded labels and sequence lengths (for each entry of the batch)
        """
        encoded = encode_sequences(sequences=gts, vocab=self.vocab, target_size=self.max_length, eos=len(self.vocab))
        seq_len = [len(word) for word in gts]
        return encoded, seq_len


class RecognitionPostProcessor(NestedObject):
    """Abstract class to postprocess the raw output of the model

    Args:
    ----
        vocab: string containing the ordered sequence of supported characters
    """

    def __init__(
        self,
        vocab: str,
        blacklist: Optional[List[str]] = None,
    ) -> None:
        self.vocab = vocab

        # Check that the blacklist is valid: list of characters only 1 char long
        if blacklist and not all(isinstance(c, str) and len(c) == 1 for c in blacklist):
            raise ValueError("Blacklist must be a list of characters")

        self.blacklist = blacklist
        self._embedding = list(self.vocab) + ["<eos>"]

    def extra_repr(self) -> str:
        return f"vocab_size={len(self.vocab)}"
