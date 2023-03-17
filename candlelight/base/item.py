
from dataclasses import dataclass
from typing import List
from torch import Tensor

@dataclass
class Item:
  source: str | None = None
  split: str | None = None
  labels: List[str] | str | None = None
  tensor: Tensor | None = None
  