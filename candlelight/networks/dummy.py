import numpy as np
from torch import Tensor
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from dataclasses import dataclass


class DummyData:
  
  def __init__(
    self,
    length: int = 20,
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    train_length: int = 20,
    test_length: int = 20,
    val_length: int = 20,
    inference_length: int = 20,
    input_length: int = 5,
    dtype=torch.double,
  ) -> None:
    self.length: int = length
    self.device: torch.device = device
    self.dtype: torch.dtype = dtype
    
    # Train data
    self.train_x: Tensor = self.random_tensor((train_length, input_length))
    self.train_y: Tensor = self.get_y(self.train_y)
    self.train_dataset:TensorDataset = TensorDataset(
      self.train_x,
      self.train_y,
    )

  @staticmethod
  def f(x: Tensor) -> Tensor:
      """Create dummy target values for a given input tensor x"""
      return x + x**2 - torch.log(x) + torch.sin(x)
    
  def random_tensor(self, shape: tuple):
    """generate random data for a given length"""
    return Tensor([i for i in torch.randn(shape, dtype=torch.double, device =self.device)], device=self.device, dtype=torch.double)
  
  def get_y(self, x: Tensor):
    """generate random data for a given length"""
    return Tensor([DummyData.f(i) for i in x], device=self.device, dtype=torch.double)
