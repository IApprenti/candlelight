from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Tuple, TypeVar
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from .. import constants as c

Value = TypeVar('Value', str, int, float, bool)
Tensors = TypeVar('Tensors', Tensor, Dict[str, Tensor], Tuple[Tensor,...])
Names = TypeVar('Names', str, List[str], Tuple[str,...])

class Classics(NamedTuple):
  encodings: str | List[str] = c.ENCODINGS
  targets: str | List[str] = c.TARGETS
  predictions: str | List[str] = c.PREDICTIONS
  embeddings: str | List[str] = c.EMBEDDINGS
  mappings: str | List[str] = c.MAPPINGS
  weights: str | List[str] = c.WEIGHTS
  classes: str | List[str] = c.CLASSES
  scores: str | List[str] = c.SCORES
  
class Item:
  pass

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
    tensors: Dict[str, Tensor] | None = None,
    classics: Classics | None = None,
    labels: Tuple[str,...] | None = None,
    current_index: int = 0,
    info: Dict[str, Any] | str | None = None,
    **kwargs : Dict[str, Any],
  ):
    self.root_directory: str | None = root_directory
    self.sources: List[str] | None = sources
    self.splits: List[str] | None = splits
    self.tensors: Dict[str, Tensor] = tensors if tensors is not None else {}
    self.classics: Classics | None = classics if classics is not None else Classics()
    self.labels: Tuple[str,...] | None = labels
    self.current_index: int = current_index
    self.info: Dict[str, Any] | str | None = info
    
    for key, value in kwargs.items():
      setattr(self, key, value)
    
  ###  Named constructor  ###
  
  @classmethod
  def from_classics(
    cls,
    root_directory: str | None = None,
    sources: List[str] | None = None,
    splits: List[str] | None = None,
    tensors_names: Classics | None = None,
    encodings: Tensors | None = None,
    targets: Tensors | None = None,
    predictions: Tensors | None = None,
    embeddings: Tensors | None = None,
    mappings:  Tensors | None = None,
    weights:  Tensors | None = None,
    scores:  Tensors | None = None,
    labels: Tuple[str,...] | None = None,
    current_index: int = 0,
    info: Dict[str, Any] | str | None = None,
    **kwargs : Dict[str, Tensors],
  ) -> Items:
    """Return a new instance of the class."""
    # TODO: this needs an implementation
    
    tensors: Dict[str, Tensor] = {}
    
    
  ##############################
  ###   Tensors management   ###
  ##############################
  
  @property
  def defined_tensors(self) -> List[str]:
    """Return the list of defined tensors."""
    return list(self.tensors.keys())
  
  def tensor_exists(self, name: str) -> bool:
    """Return True if the tensor exists."""
    return name in self.defined_tensors
  
  def tensor_has_index(self, name: str, index: int) -> bool:
    """Return True if the tensor has the given index."""
    return self.tensor_exists(name) and index < len(self.tensors[name])
  
  ###   Get a tensor from the dict   ###
  
  # Single tensor
  
  def get_tensor(self, name: str | None) -> Tensor | None:
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
  
  # Dict of tensors
  
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
        
  # Tuple of tensors
  
  def get_tensors_as_tuple(self, names: Tuple[str,...] | List[str]) -> Tuple[Tensor] | None:
    """Return the tensors with the given names."""
    keys = [key for key in names if key in self.defined_tensors]
    if len(keys) == 0:
      return None
    return (self.tensors[key] for key in keys)
  
  def set_tensors_as_tuple(self, names: Tuple[str,...] | List[str], tensors: Tuple[Tensor, ...]) -> None:
    """Set the tensor with the given name."""
    for key, tensor in zip(names, tensors):
      self.tensors[f'{key}'] = tensor
        
  # Single tensor or dict of tensors
  
  def get_one_or_many_tensors(self, names: Names | None) -> Tensors | None:
    """Return one or many tensors."""
    if names is None:
      return None
    if isinstance(names, str):
      return self.get_tensor(names)
    return self.get_tensors(names)
  
  def set_one_or_many_tensors(self, tensors: Tensors | None, names: Names | None) -> None:
    """Set one or many tensors."""
    if names is None or tensors is None:
      return
    if isinstance(names, str):
      self.set_tensor(tensors, names)
    self.set_tensors(tensors)
    
  def del_one_or_many_tensors(self, names: Names | None) -> None:
    """Delete one or many tensors."""
    if names is None:
      return
    if isinstance(names, str):
      self.del_tensor(names)
    self.del_tensor(names)
    
  ###   Get a tensor from the dict at index   ###
  
  # Single tensor
  
  def get_tensor_at_index(self, name: str | None, index: int | None = None) -> Tensor:
    """Return the encoding tensor at the given index."""
    if name is None:
      return None
    if index is None:
      index = self.current_index
    if not self.tensor_has_index(name, index):
      return None
    return self.tensors[name][index]
  
  def set_tensor_at_index(self, tensor: Tensor, name: str | None, index: int | None = None) -> None:
    """Set the encoding tensor at the given index."""
    if name is None:
      return
    if index is None:
      index = self.current_index
    if not self.tensor_has_index(name, index):
      return
    self.tensors[name][index] = tensor
    
  def del_tensor_at_index(self, name: str | None, index: int | None = None) -> None:
    """Delete the encoding tensor at the given index."""
    if name is None:
      return
    if index is None:
      index = self.current_index
    if not self.tensor_has_index(name, index):
      return
    del self.tensors[name][index]
    
  # Dict of tensors
  
  def get_tensors_at_index(self, names: List[str] | None = None, index: int | None = None,) -> Dict[str, Tensor]:
    """Return the encoding tensor at the given index."""
    if names is None:
      names = self.defined_tensors
    if len(names) == 0:
      return None
    if index is None:
      index = self.current_index
    result: Dict[str, Tensor] = {}
    for name in names:
      value = None
      if self.tensor_has_index(name, index):
        value = self.tensors[name][index]
      result[name] = value
    return result
  
  def set_tensors_at_index(self, tensor: Tensor, names: List[str] | None, index: int | None = None) -> None:
    """Set the encoding tensor at the given index."""
    if names is None:
      names = self.defined_tensors
    if len(names) == 0:
      return None
    if index is None:
      index = self.current_index
    for name in names:
      if self.tensor_has_index(name, index):
        self.tensors[name][index] = tensor
    
  def del_tensors_at_index(self, names: List[str] | None, index: int | None = None) -> None:
    """Delete the encoding tensor at the given index."""
    if names is None:
      names = self.defined_tensors
    if len(names) == 0:
      return None
    if index is None:
      index = self.current_index
    for name in names:njut
      if self.tensor_has_index(name, index):
        del self.tensors[name][index]
  
  ############################
  ###   Specific tensors   ###
  ############################
  
  ###   Encodings   ###
  
  @property
  def encodings(self) -> Tensors | None:
    """Return the encodings tensor(s)."""
    return self.get_one_or_many_tensors(self.tensors_names.encodings)
  
  @encodings.setter
  def encodings(self, tensors: Tensors) -> None:
    """Set the encodings tensor(s)."""
    self.set_one_or_many_tensors(tensors, self.tensors_names.encodings)
    
  @encodings.deleter
  def encodings(self) -> None:
    """Delete the encodings tensor(s)."""
    self.del_one_or_many_tensors(self.tensors_names.encodings)
    
  ###   Targets   ###
  
  @property
  def targets(self) -> Tensors | None:
    """Return the targets tensor(s)."""
    return self.get_one_or_many_tensors(self.tensors_names.targets)
  
  @targets.setter
  def targets(self, tensors: Tensors) -> None:
    """Set the targets tensor(s)."""
    self.set_one_or_many_tensors(tensors, self.tensors_names.targets)
    
  @targets.deleter
  def targets(self) -> None:
    """Delete the targets tensor(s)."""
    self.del_one_or_many_tensors(self.tensors_names.targets)
    
  ###   Predictions   ###
  
  @property
  def predictions(self) -> Tensors | None:
    """Return the predictions tensor(s)."""
    return self.get_one_or_many_tensors(self.tensors_names.predictions)
  
  @predictions.setter
  def predictions(self, tensors: Tensors) -> None:
    """Set the predictions tensor(s)."""
    self.set_one_or_many_tensors(tensors, self.tensors_names.predictions)
    
  @predictions.deleter
  def predictions(self) -> None:
    """Delete the predictions tensor(s)."""
    self.del_one_or_many_tensors(self.tensors_names.predictions)
    
  ###   mappings   ###
  
  @property
  def mappings(self) -> Tensors | None:
    """Return the mappings tensor(s)."""
    return self.get_one_or_many_tensors(self.tensors_names.mappings)
  
  @mappings.setter
  def mappings(self, tensors: Tensors) -> None:
    """Set the scores tensor(s)."""
    self.set_one_or_many_tensors(tensors, self.tensors_names.mappings)
    
  @mappings.deleter
  def mappings(self) -> None:
    """Delete the mappings tensor(s)."""
    self.del_one_or_many_tensors(self.tensors_names.mappings)
    
  ###   weights   ###
  
  @property
  def weights(self) -> Tensors | None:
    """Return the weights tensor(s)."""
    return self.get_one_or_many_tensors(self.tensors_names.weights)
  
  @weights.setter
  def weights(self, tensors: Tensors) -> None:
    """Set the weights tensor(s)."""
    self.set_one_or_many_tensors(tensors, self.tensors_names.weights)
    
  @weights.deleter
  def weights(self) -> None:
    """Delete the weights tensor(s)."""
    self.del_one_or_many_tensors(self.tensors_names.weights)
    
  ###   Embeddings   ###
  
  @property
  def embeddings(self) -> Tensors | None:
    """Return the embeddings tensor(s)."""
    return self.get_one_or_many_tensors(self.tensors_names.embeddings)
  
  @embeddings.setter
  def embeddings(self, tensors: Tensors) -> None:
    """Set the embeddings tensor(s)."""
    self.set_one_or_many_tensors(tensors, self.tensors_names.embeddings)
    
  @embeddings.deleter
  def embeddings(self) -> None:
    """Delete the embeddings tensor(s)."""
    self.del_one_or_many_tensors(self.tensors_names.embeddings)
    
  ###   Scores   ###
  
  @property
  def scores(self) -> Tensors | None:
    """Return the scores tensor(s)."""
    return self.get_one_or_many_tensors(self.tensors_names.scores)
  
  @embeddings.setter
  def scores(self, tensors: Tensors) -> None:
    """Set the scores tensor(s)."""
    self.set_one_or_many_tensors(tensors, self.tensors_names.scores)
    
  @embeddings.deleter
  def scores(self) -> None:
    """Delete the scores tensor(s)."""
    self.del_one_or_many_tensors(self.tensors_names.scores)
    
  ###   Classes   ###
  
  @property
  def classes(self) -> Tensors | None:
    """Return the classes tensor(s)."""
    return self.get_one_or_many_tensors(self.tensors_names.classes)
  
  @embeddings.setter
  def classes(self, tensors: Tensors) -> None:
    """Set the classes tensor(s)."""
    self.set_one_or_many_tensors(tensors, self.tensors_names.classes)
    
  @embeddings.deleter
  def classes(self) -> None:
    """Delete the classes tensor(s)."""
    self.del_one_or_many_tensors(self.tensors_names.classes)
  
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
    info = {"index": index, **self.info} if isinstance(self.info, dict) else info_str
    return Item(
      root_directory = self.root_directory,
      source = self.sources[index] if self.sources is not None else None,
      split = self.splits[index] if self.splits is not None else None,
      tensors = self.get_tensors_at_index(index),
      info = info,
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