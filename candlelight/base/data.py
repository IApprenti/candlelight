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

class Data:
  """Handle train, test, validation datasets and dataloaders.
  Store the metadata in a Pandas dataframe.
  Can save the embeddings in a monolithic tensor.
  """

  def __init__(
    self,
    root_directory: str | None = None,
    sources: List[str] | None = None,
    splits: List[str] | None = None,
    entries: Tensor | None = None,
    scores: Tensor | None = None,
    classes: Tensor | None = None,
    entries_names: List[str] | None = None,
    results_names: List[str] | None = None,
    results_tasks: List[str] | None = None,
    embeddings: Tensor | None = None,
    maps: Tensor | None = None,
    reconstructions: Tensor | None = None,
    info: Dict[str, Any] | str | None = None,
    **kwargs : Dict[str, Any],
  ):
    self.root_directory: str | None = root_directory
    self.sources: List[str] | None = sources
    self.splits: List[str] | None = splits
    self.entries: Tensor | None = entries
    self.targets: Tensor | None = targets
    self.results: Tensor | None = results
    self.entries_names: List[str] | None = entries_names
    self.results_names: List[str] | None = results_names
    self.results_tasks: List[str] | None = results_tasks
    self.embeddings: Tensor | None = embeddings
    self.maps: Tensor | None = maps
    self.reconstructions: Tensor | None = reconstructions
    self.info: Dict[str, Any] | str | None = info
    
    # Set any additional attributes.
    for key, value in kwargs.items():
      setattr(self, key, value)
    

  ###################
  ###   Getters   ###
  ###################
  
  def split(self, index: int) -> str | None:
    """Return the source at the given index."""
    if self.splits is None:
      return None
    if self.splits[0] < index + 1:
      return None
    return self.splits[index]
  
  def source(self, index: int) -> str | None:
    """Return the source at the given index."""
    if self.sources is None:
      return None
    if self.sources[0] < index + 1:
      return None
    return self.sources[index]
  
  @staticmethod
  def at_index(index: int, tensor: Tensor | None) -> Tensor | None:
    """Return the tensor at the given index."""
    if tensor is None:
      return None
    if tensor[0] < index + 1:
      return None
    return tensor[index]
  
  def entry(self, index: int) -> Tensor | None:
    """Return the data at index."""
    return Data.at_index(index, self.entries)
  
  def target(self, index: int) -> Tensor | None:
    """Return the data at index."""
    return Data.at_index(index, self.targets)
  
  def result(self, index: int) -> Tensor | None:
    """Return the data at index."""
    return Data.at_index(index, self.results)
  
  def embedding(self, index: int) -> Tensor | None:
    """Return the data at index."""
    return Data.at_index(index, self.embeddings)
  
  def map(self, index: int) -> Tensor | None:
    """Return the data at index."""
    return Data.at_index(index, self.maps)
  
  def reconstruction(self, index: int) -> Tensor | None:
    """Return the data at index."""
    return Data.at_index(index, self.reconstructions)
  
  def item(self, index: int) -> Item | None:
    """Return the data at index."""
    return Item(
      source=self.source(index),
      split=self.split(index),
      tensor=self.entry(index),
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
    return Data(
      
    )
    
  def __setitem__(self, index: int, value: Item | NotImplementedError):
    """Set the data at the given index."""
    pass
  
  def __delitem__(self, index: int):
    """Delete the data at the given index."""
    pass
  
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