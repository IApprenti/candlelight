from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

from .. import constants as c

from .item import Item

Value = TypeVar('Value', str, int, float, bool)

class Items:
  """Save the various elements used during an AI pipeline
  """

  def __init__(
    self,
    root_directory: str | None = None,
    sources: List[str] | None = None,
    splits: List[str] | None = None,
    encodings: Tensor | None = None,
    scores: Tensor | None = None,
    scores_targets: Tensor | None = None,
    scores_names: List[str] | None = None,
    classes: Tensor | None = None,
    classes_targets: Tensor | None = None,
    classes_names: List[str] | None = None,
    embeddings: Tensor | None = None,
    heatmaps: Tensor | None = None,
    reconstructions: Tensor | None = None,
    info: Dict[str, Any] | str | None = None,
    **kwargs : Dict[str, Any],
  ):
    self.root_directory: str | None = root_directory
    self.sources: List[str] | None = sources
    self.splits: List[str] | None = splits
    self.encodings: Tensor | None = encodings
    self.scores: Tensor | None = scores
    self.scores_targets: Tensor | None = scores_targets
    self.scores_names: List[str] | None = scores_names
    self.classes: Tensor | None = classes
    self.classes_targets: Tensor | None = classes_targets
    self.classes_names: List[str] | None = scores_names
    self.embeddings: Tensor | None = embeddings
    self.heatmaps: Tensor | None = heatmaps
    self.reconstructions: Tensor | None = reconstructions
    self.info: Dict[str, Any] | str | None = info
    
    # Set any additional attributes.
    for key, value in kwargs.items():
      setattr(self, key, value)
  
  #######################
  ###   Create from   ###
  #######################
  
  @classmethod
  def from_classes_images(cls, path: str) -> Items | None:
    """Create items from a directory of images arranged per class"""
    pass
  
  @classmethod
  def from_dataframe(cls, source: pd.DataFrame) -> Items | None:
    """Create items from a dataframe"""
    pass
  
  @classmethod
  def from_csv(cls, source: str) -> Items | None:
    """Create Items from a csv file"""
    df: pd.DataFrame = ...
    return cls.from_dataframe(df)
  
  @classmethod
  def from_google_sheet(cls, source: str) -> Items | None:
    """Create items from a google sheet"""
    df: pd.DataFrame = ...
    return cls.from_dataframe(df)
  
  #################################
  ###   Torch data management   ###
  #################################
  
  def dataset(self, split: str | None = None) -> Dataset:
    pass
  
  def dataloader(self, split: str | None = None) -> Dataloader:
    pass
  
  ###################
  ###   Getters   ###
  ###################
  
  @staticmethod
  def at_tensor_index(index: int, arr: Tensor | None) -> Tensor | None:
    """Return the item value at the given index."""
    if arr is None:
      return None
    if arr[0] < index + 1:
      return None
    return arr[index]
  
  @staticmethod
  def at_list_index(index: int, arr: List[Value] | None) -> Value | None:
    """Return the item value at the given index."""
    if arr is None:
      return None
    if arr[0] < index + 1:
      return None
    return arr[index]
  
  def item(self, index: int) -> Item | None:
    """Return the data at index."""
    return Item(
      source=self.list_index(index,self.sources),
      split=self.at_list_index(index,self.splits),
      encodings=self.at_tensor_index(index, self.encodings),
    )
  
  #########################
  ###   Dunder methods  ###
  #########################
  
  def __str__(self):
    """Return the string representation."""
    if self.info is not None:
      return f"Data: {self.info}"
    return f"Data of length {len(self)}."
  
  def __repr__(self) -> str:
    return str(self)
  
  def __len__(self) -> int:
    """Return the number of data points."""
    if self.entries is not None:
      return self.entries.shape[0]
    if self.sources is not None:
      return len(self.sources)
    if self.embeddings is not None:
      return self.embeddings.shape[0]
    return 0
  
  def __getitem__(self, index: int) -> Item | NotImplementedError:
    """Return the data at the given index."""
    return self.item(index)
    
  def __setitem__(self, index: int, value: Item | NotImplementedError):
    """Set the data at the given index."""
    self.set_item(value)
  
  def __delitem__(self, index: int):
    """Delete the data at the given index."""
    raise NotImplemented
  
  def __iter__(self):
    """Return an iterator over the data."""
    pass
  
  def __contains__(self, item: Item | NotImplementedError) -> bool:
    """Return whether the data contains the given item."""
    pass
  
  def __add__(self, other: Data) -> Data:
    """Return the concatenation of the data."""
    pass
  
  def __iadd__(self, other: Data) -> Data:
    """Return the concatenation of the data."""
    pass
