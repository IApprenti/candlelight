from __future__ import annotations
from typing import Any, Dict
from torch import Tensor
import pickle

class Item:
    """Store a single item of data.
    Can be inherited to create custom items such as images, text embeddings, etc.
    """

    def __init__(
            self,
            root_directory: str | None = None,
            split: str | None = None,
            source: str | None = None,
            encoding: Tensor | None = None,
            target: Tensor | Dict[str,Tensor] | None = None,
            index: int | None = None,
            info: dict[str, str] | str | None = None,
            **kwargs: Dict[str, Any]
    ) -> None:        
        self.root_directory: str | None = root_directory
        self.split: str | None = split
        self.source: str | None = source
        self._encoding: Tensor | None = encoding
        self.target: Tensor | Dict[str,Tensor] | None = target
        self.index: int | None = index
        self.info: dict[str, str] | str | None = info

        for key, value in kwargs.items():
          setattr(self, key, value)

    ####################
    ###   Metadata   ###
    ####################

    def __str__(self) -> str:
        """Return a string representation of the item."""
        if self.source is not None:
            return f'{self.__class__.__name__}({self.source})'
        if self.encoding is not None:
            return f'{self.__class__.__name__}({self.encoding.shape})'
        return f'{self.__class__.__name__}()'
    
    def __len__(self) -> int:
        """Return the length of the item.
        """
        if self.encoding is not None:
            return self.encoding.shape[0]
        return 1

    def shape(self) -> tuple[int, ...]:
        """Return the shape of the item.
        """
        if self.encoding is not None:
            return self.encoding.shape
        return ()

    ####################
    ###   Encoding   ###
    ####################

    @property
    def encoding(self) -> Tensor | None:
        if self._encoding is None:
            self._encoding = self.encode()
        return self._encoding
        
    @encoding.setter
    def encoding(self, encoding: Tensor) -> None:
        self._encoding = encoding

    @encoding.deleter
    def encoding(self) -> None:
        self._encoding = None
    
    def encode(self) -> Tensor | None:
        """Encode the item.
        Need to be implemented in the child class.
        """
        return None
    
    ################
    ###   Show   ###
    ################

    def show(self) -> None:
        """Show the item.
        """
        print(self)

    ################
    ###   Save   ###
    ################

    # Use Pickle by default, can be replaced by other methods

    def save(self, path: str) -> None:
        """Save the item.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> Item:
        """Load the item.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)