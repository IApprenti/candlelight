from typing import Dict, List
from typing import NamedTuple
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset, Dataloader

class Names(NamedTuple):
  """Tensors names"""
  entries = "entries"
  

class Data(Dataset):
    def __init__(
      self,
      dataframe: pd.DataFrame | None = None,
      tensors: Dict[str, Tensor] | None = None,
      names: Names = Names(),
    ):
        
        self.df = dataframe if dataframe is not None else pd.Dataframe()
        self.tensors = tensors if tensors is not None else {}
        
    @property
    def available_tensors(self) -> List[str]:
      return self.tensors.keys()
    
    ###   Setters / Getters for tensors   ###
    
    def set_tensor(self, name: str, tensor: Tensor) -> None:
        """Set the value for a tensor"""
        self.tensor[name] = tensor
        
    def get_tensor(self, name: str) -> Tensor | None:
        """Get the value for a tensor"""
        if not name in self.available_tensors:
            return None
        return self.tensors[name]
      
    def del_tensor(self, name: str) -> None:
        """Delete a tensor"""
        if not name in self.available_tensors:
            return
        del self.tensors[name]
    
    ###   Entries   ###
    
    @property
    def entries(self) -> Tensor | None :
        """Get the tensor"""
        return self.get_tensor(self.names.entries)
     
    @entries.setter
    def entries(self, value: Tensor) -> None:
        """Set the tensor"""
        self.set_tensor(self.names.entries,value)
        
    @entries.deleter:
    def entries(self, value: Tensor) -> None:
        """Delete the tensor"""
        self.del_tensor(self.names.entries)
  
