from typing import Any, Dict
from torch import Tensor

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
            column: str | None = None,
            index: int | None = None,
            info: dict[str, str] | str | None = None,
            **kwargs: Dict[str, Any]
    ) -> None:        
        self.root_directory: str | None = root_directory
        self.split: str | None = split
        self.source: str | None = source
        self._encoding: Tensor | None = encoding
        self.column: str | None = column
        self.index: int | None = index
        self.info: dict[str, str] | str | None = info

        for key, value in kwargs.items():
          setattr(self, key, value)

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
        """
        return None