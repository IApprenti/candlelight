from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, NamedTuple, TypeVar
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from .. import constants as c
from .item import Item

Value = TypeVar('Value', str, int, float, bool)
Tensors = TypeVar('Tensors', Tensor, Dict[str, Tensor])
Names = TypeVar('Names', str, List[str])

class TensorNames(NamedTuple):
  encodings: str | List[str] = c.ENCODINGS
  targets: str | List[str] = c.TARGETS
  predictions: str | List[str] = c.PREDICTIONS
  embeddings: str | List[str] = c.EMBEDDINGS
  mappings: str | List[str] = c.MAPPINGS
  weights: str | List[str] = c.WEIGHTS

class Items:
  """Save the various elements used during an AI pipeline.
  It can be used to store a batch of items or a full dataset.
  It is meant to be inherited from and modified to fit the needs of the project.
  """

  def __init__(
    self,
    root_directory: str | None = None,
    sources: List[str] | None = None,
    splits: List[str] | None = None,
    encodings: Tensors | None = None,
    targets: Tensors | None = None,
    predictions: Tensors | None = None,
    embeddings: Tensors | None = None,
    mappings:  Tensors | None = None,
    weights:  Tensors | None = None,
    scores:  Tensors | None = None,
    labels: List[str] | None = None,
    info: Dict[str, Any] | str | None = None,
    **kwargs : Dict[str, Any],
  ):
    self.root_directory: str | None = root_directory
    self.sources: List[str] | None = sources
    self.splits: List[str] | None = splits
    self.labels: List[str] | None = labels
    self.info: Dict[str, Any] | str | None = info
    
    # Set any additional attributes.
    for key, value in kwargs.items():
      setattr(self, key, value)
      
    # Initialize the tensors dict
    
    self.tensors: Dict[str, Tensor] | None = self.initialize_tensors(
      encodings = encodings,
      targets = targets,
      predictions = predictions,
      embeddings = embeddings,
      mappings = mappings,
      weights = weights,
      scores = scores,
    )
    
  ##############################
  ###   Tensors management   ###
  ##############################
  
  @property
  def defined_tensors(self) -> List[str]:
    """Return the list of defined tensors."""
    return list(self.tensors.keys())
  
  ###   Single tensor   ###
  
  def get_tensor(self, name: str) -> Tensor | None:
    """Return the tensor with the given name."""
    if name in self.defined_tensors:
      return self.tensors[name]
    return None
  
  def set_tensor(self, tensor: Tensor, name: str) -> None:
    """Set the tensor with the given name."""
    self.tensors[name] = tensor
    
  def del_tensor(self, name: str) -> None:
    """Delete the tensor with the given name."""
    if name in self.defined_tensors:
      del self.tensors[name]
  
  ###   Multiple tensors   ###
  
  def get_tensors(self, names: List[str]) -> Dict[Tensor] | None:
    """Return the tensors with the given names."""
    keys = [key for key in names if key in self.defined_tensors]
    if len(keys) == 0:
      return None
    return {key: self.tensors[key] for key in keys}
  
  def set_tensors(self, tensors: Dict[str, Tensor]) -> None:
    """Set the tensor with the given name."""
    for key, tensor in tensors.items():
      self.tensors[f'{key}'] = tensor
    
  def del_tensor(self, names: List[str]) -> None:
    """Delete the tensor with the given name."""
    for name in names:
      if name in self.defined_tensors:
        del self.tensors[name]
        
  ###   Flexible tensors   ###
  
  def get_one_or_many_tensors(self, names: Names) -> Tensors | None:
    """Return one or many tensors."""
    if isinstance(names, str):
      return self.get_tensor(names)
    return self.get_tensors(names)
  
  def set_one_or_many_tensors(self, tensors: Tensors, names: Names | None) -> None:
    """Set one or many tensors."""
    if isinstance(names, str):
      self.set_tensor(tensors, names)
    self.set_tensors(tensors)
    
  def del_one_or_many_tensors(self, names: Names) -> None:
    """Delete one or many tensors."""
    if isinstance(names, str):
      self.del_tensor(names)
    self.del_tensor(names)
  
  ############################
  ###   Specific tensors   ###
  ############################
  
  ###   Encodings   ###
  
  @property
  def encodings(self) -> Tensors | None:
    """Return the encodings tensor."""
    return self.get_one_or_many_tensors(TensorNames.encodings)
  
  @encodings.setter
  def encodings(self, tensors: Tensors) -> None:
    """Set the encodings tensor."""
    self.set_one_or_many_tensors(tensors, TensorNames.encodings)
    
  @encodings.deleter
  def encodings(self) -> None:
    """Delete the encodings tensor."""
    self.del_one_or_many_tensors(TensorNames.encodings)
    
  ###   Targets   ###
  
  @property
  def targets(self) -> Tensors | None:
    """Return the targets tensor."""
    return self.get_one_or_many_tensors(TensorNames.targets)
  
  @targets.setter
  def targets(self, tensors: Tensors) -> None:
    """Set the targets tensor."""
    self.set_one_or_many_tensors(tensors, TensorNames.targets)
    
  @targets.deleter
  def targets(self) -> None:
    """Delete the targets tensor."""
    self.del_one_or_many_tensors(TensorNames.targets)
  
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
  
  def dataloader(self, split: str | None = None) -> DataLoader:
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
      root_directory = xxx,
      source = xxx,
      split = xxx,
      encoding = xxx,
      prediction = xxx,
      target = xxx,
      labels = xxx,
      info = xxx,
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


class Item:
  pass

class Data(Dataset):
  
  def __init__(
    self,
    items: Items,
  ) -> None:
    self.items: Items = items
    
  def __len__(self) -> int:
    return len(self.items)
  
  def __getitem__(self, index: int) -> Item:
    return self.items[index]
  
  def dataset(self, split: str | None = None) -> Dataset:
    return self
  
  def dataloader(self, split: str | None = None) -> DataLoader:
    return DataLoader(self)