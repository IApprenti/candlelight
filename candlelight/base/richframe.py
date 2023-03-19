"""This is a rich dataframe that can store tensors and other data types.

The idea is to have a table of items:
- Each column store a value (physically stored in a dataframe) or a tensor (physically stored in a dict of tensors).
- Each line store an item (physically stored in a dataframe or a tensor).

It is physically stored in a dataframe and a dict of tensors while this duality is hidden to the user.

WORK IN PROGRESS
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Tuple, TypeVar
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from .. import constants as c

AValueType =  str | float | int | bool | Tensor
Value = TypeVar('Value', str, float, int, bool, Tensor)
Values = TypeVar('Values', List[str], List[float], List[int], List[bool], Tensor)
Column = List[Value]
Line = List[AValueType]
Tensors = Dict[str, Tensor]
Tensor_s = TypeVar('Tensor_s', Tensor, Tensors)

  

class RichFrame:
  """Save the various elements used during an AI pipeline.
  It can be used to store a batch of items or a full dataset.
  Beware: The dataframe and the tensor shall not have the same columns.
  """

  def __init__(
    self,
    dataframe: pd.DataFrame | None = None,
    tensors: Dict[str, Tensor] | None = None,
    root_directory: str | None = None,
    current_index: int = 0,
    info: Dict[str, Any] | str | None = None,
    **kwargs : Dict[str, Any],
  ):
    self.root_directory: str | None = root_directory
    self.dataframe: pd.DataFrame = dataframe if dataframe is not None else pd.DataFrame()
    self.tensors: Dict[str, Tensor] = tensors if tensors is not None else {}
    self.current_index: int = current_index
    self.info: Dict[str, Any] | str | None = info
    
    for key, value in kwargs.items():
      setattr(self, key, value)

  #################
  ###   Slice   ###
  #################

  def at(
      self,
      indexs: int | List[int],
      columns: str | List[str]
    ) -> RichFrame:
    """Slice the dataframe and the tensors at the given indexes and columns."""
    if isinstance(indexs, int):
      indexs = [indexs]
    if isinstance(columns, str):
      columns = [columns]
    dataframe = self.dataframe.iloc[indexs, columns]
    tensors = {column: self.tensors[column][indexs] for column in columns if column in self.tensors.keys()}
    return RichFrame(dataframe=dataframe, tensors=tensors)

  #############################
  ###   To and from lists   ###
  #############################

  ###   Column   ###

  def get_column(self, column: str) -> List[Value] | Tensor | None:
    """Return the given column as a list."""
    if column in self.dataframe.columns:
      return self.dataframe[column].tolist()
    elif column in self.tensors.keys():
      return self.tensors[column]
    return None

  def set_column(self, column: str, values: Values) -> None:
    """Set the given column with the given values."""
    if isinstance(values, Tensor):
      self.tensors[column] = values
    elif isinstance(values, list) and isinstance(values[0], Tensor):        
      self.tensors[column] = torch.stack(values)
    else:   
      self.dataframe[column] = values

  def del_column(self, column: str) -> bool:
    """Delete the given column."""
    if column in self.tensors.keys():
      del self.tensors[column]
      return True
    elif column in self.dataframe.columns:
      del self.dataframe[column]
      return True
    return False

  ###   Line   ###

  def get_line(self, index: int) -> Line | None:
    """Return the given line as a Series."""
    if index not in self.dataframe.index:
      return None
    dataframe_line = self.dataframe.iloc[index].tolist()
    tensor_line = [self.tensors[column][index] for column in self.tensors.keys()]
    return dataframe_line + tensor_line

  def set_line(self, index: int, values: Values) -> None:
    """Set the given line with the given values."""
    dataframe_values = values[:len(self.dataframe.columns)]
    self.dataframe.iloc[index] = pd.Series(dataframe_values)
    tensor_values = values[len(self.dataframe.columns):]
    for column, value in zip(self.tensors.keys(), tensor_values):
      self.tensors[column][index] = value

  def del_line(self, index: int) -> None:
    """Delete the given line."""
    if index not in self.dataframe.index:
      return
    self.dataframe.drop(index, inplace=True)
    for column in self.tensors.keys():
      self.tensors[column] = torch.cat((self.tensors[column][:index], self.tensors[column][index+1:]))

  def append_line(self, values: Values) -> None:
    """Append the given values as a new line."""
    dataframe_values = values[:len(self.dataframe.columns)]
    self.dataframe.loc[len(self.dataframe)] = dataframe_values
    tensor_values = values[len(self.dataframe.columns):]
    for column, value in zip(self.tensors.keys(), tensor_values):
      self.tensors[column] = torch.cat((self.tensors[column], value.unsqueeze(0)))

  ###   Value   ###

  def get_value(self, column: str, index: int | None = None) -> AValueType | None:
    """Return the value of the given column at the given index."""
    if index is None:
      index = self.current_index
    if column in self.dataframe.columns:
      return self.dataframe[column][index]
    elif column in self.tensors.keys():
      return self.tensors[column][index]
    return None
  
  def set_value(self, column: str, value: AValueType, index: int | None = None) -> None:
    """Set the value of the given column at the given index."""
    if index is None:
      index = self.current_index
    if column in self.tensors.keys():
      self.tensors[column][index] = value
    else:
      self.dataframe[column][index] = value

  ###   Batch   ###

  def get_batch(self, indices: List[int]) -> Items:
    """Return the given columns at the given indices as a Batch."""
    batch: Items = Items()
    for column in self.dataframe.columns:
      batch.set_column(column, self.get_column(column)[indices])
    for column in self.tensors.keys():
      batch.set_column(column, self.get_column(column)[indices])
    return batch
  
  def set_batch(self, batch: Items, indices: List[int]) -> None:
    """Set the given columns at the given indices with the given values."""
    for column in batch.dataframe.columns:
      self.set_column(column, batch.get_column(column)[indices])
    for column in batch.tensors.keys():
      self.set_column(column, batch.get_column(column)[indices])

  def append_batch(self, batch: Items) -> None:
    """Add the given batch at the end of the dataset."""
    self.set_batch(batch, list(range(len(self), len(self) + len(batch))))

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
    
  def del_tensors(self, names: List[str]) -> None:
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