"""User-defined types."""

from __future__ import annotations

from typing import Dict, Tuple

from torch import Tensor

DataGroup = Tuple[Tensor, Tensor]
LossDict = Dict[str, Tensor]
