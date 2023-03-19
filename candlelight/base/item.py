from __future__ import annotations
from typing import Any, Dict
from torch import Tensor
import pickle
from copy import deepcopy
import numpy as np
import PIL.Image
import os
import torch

import torchvision.transforms as transforms

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
            predictions: Tensor | Dict[str,Tensor] | None = None,
            tensors: Tensor | Dict[str,Tensor] | None = None,
            transforms: transforms.Compose | None = None,
            index: int | None = None,
            info: dict[str, str] | str | None = None,
            **kwargs: Dict[str, Any]
    ) -> None:        
        self.root_directory: str | None = root_directory
        self.split: str | None = split
        self.source: str | None = source
        self._encoding: Tensor | None = encoding
        self.target: Tensor | Dict[str,Tensor] | None = target
        self.predictions: Tensor | Dict[str,Tensor] | None = predictions
        self.tensors: Tensor | Dict[str,Tensor] | None = tensors
        self.transforms: transforms.Compose | None = transforms
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

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the item.
        """
        if self.encoding is not None:
            return self.encoding.shape
        return ()
    
    def copy(self) -> Item:
        """Return a copy of the item.
        """
        return deepcopy(self)
    
    def __call__(self) -> Any:
        return self.encoding

    @property
    def path(self) -> str | None:
        """Return the path of the item.
        """
        if self.root_directory is not None and self.source is not None:
            if not self.root_directory in self.source:
                return os.path.join(self.root_directory, self.source)
        if self.source is not None:
            return self.source
        return None
    
    ############################################
    ###   Convert the source into a tensor   ###
    ############################################

    def to_tensor(self, path: str | None = None) -> Tensor | None:
        """Convert the source into a tensor.
        This method is not garanteed to work for all sources.
        Reimplement it in the child class if needed.
        """
        if path is None:
            path = self.source
        if path is None:
            return None
        extension = os.path.splitext(path)[1]
        if extension in ['.pkl', '.pickle', '.p', '.pt', '.pth', '.pth.tar', '.pt.tar']:
            try:
                return torch.load(path)
            except:
                pass
        elif extension in ['.npy', '.npz']:
            try:
                return torch.from_numpy(np.load(path))
            except:
                pass
        elif extension in ['.txt', '.csv']:
            try:
                return torch.from_numpy(np.loadtxt(path))
            except:
                pass
        elif extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff']:
            try:
                image = self.load_image(path)
                if image is not None:
                    return self.image_to_tensor(image)
            except:
                pass
        return None
    

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
        encoding = self.encoding
        if encoding is not None:
            if encoding.dim() > 1:
                self.show_image()
            else:
                print(encoding)
        else:
            print(self)

    #################
    ###   Image   ###
    #################

    def image_to_tensor(self, image: PIL.Image.Image) -> Tensor | None:
        """Convert an image into a tensor.
        """
        if image is not None:
            if self.transforms is None:
                return transforms.ToTensor()(image)
            else:
                return self.transforms(image)
        return None
    
    def tensor_to_image(self, tensor: Tensor) -> PIL.Image.Image | None:
        """Convert a tensor into an image.
        """
        if tensor is not None:
            return transforms.ToPILImage()(tensor)
        return None

    def load_image(self, path: str | None = None) -> PIL.Image.Image | None:
        """Load the item as an image.
        """
        if path is None:
            path = self.path
        if path is not None:
            try:
                return PIL.Image.open(path)
            except:
                pass
        return None
    
    def save_image(self, path: str | None = None, tensor: Tensor | None = None) -> bool:
        """Save the item as an image.
        """
        if path is None:
            path = self.path
        if tensor is None:
            tensor = self.encoding
        if tensor is not None and path is not None:
            image = self.get_image(tensor)
            if image is not None:
                try:
                    image.save(path)
                    return True
                except:
                    pass
        return False

    def get_image(self, tensor: Tensor | None  = None) -> PIL.Image.Image | None:
        """get the item as an image.
        """
        if tensor is None:
            tensor = self.encoding
        if tensor is not None:
            tensor = tensor*255
            array = np.array(tensor, dtype=np.uint8)
            if np.ndim(array)==4:
                # Remove the batch dimension
                array = array[0]
            if np.ndim(array)==3:
                # Move the channel dimension to the end
                array = np.transpose(array, (1,2,0))
            return PIL.Image.fromarray(array)
        elif self.source is not None:
            try:
                return PIL.Image.open(self.source)
            except:
                pass
        return None
    
    def show_image(self, tensor: Tensor | None = None) -> None:
        """Show the item as an image.
        """
        image = self.get_image(tensor)
        if image is not None:
            image.show()
        else:
            print('No image to show')

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