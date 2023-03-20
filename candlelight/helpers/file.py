from __future__ import annotations
from typing import Any, Dict
from torch import Tensor
import pickle
from copy import deepcopy
import numpy as np
from PIL.Image import Image
import os
import torch
from ..types_alias import NumpyArray
import PIL.Image
import pandas as pd
from .. import constants as c
from PIL.Image import Image
from torchvision.transforms import ToPILImage
from . import image

### Loaders ###

def to_numpy(path: str) -> NumpyArray:
    """Convert the source into a numpy array."""
    return np.load(path)

def to_image(path: str) -> Image:
    """Convert the source into an image.
    """
    return PIL.Image.open(path)

def to_dataframe(path:str) -> pd.DataFrame:
    """Convert the source into a pandas dataframe.
    """
    ext = extension(path)
    if ext in c.PICKLE_EXTENSIONS:
        return pd.read_pickle(path)
    if ext in c.FEATHER_EXTENSIONS:
        return pd.read_feather(path)
    if ext in c.PARQUET_EXTENSIONS:
        return pd.read_parquet(path)
    if ext in c.HDF_EXTENSIONS:
        return pd.read_hdf(path)
    if ext == '.json':
        return pd.read_json(path)
    if ext == '.csv':
        return pd.read_csv(path)
    if ext == '.tsv':
        return pd.read_csv(path, sep='\t')
    if ext in c.EXCEL_EXTENSIONS:
        return pd.read_excel(path)
    if ext in c.NUMPY_EXTENSIONS:
        return pd.DataFrame(to_numpy(path))
    raise ValueError(f'Unknown file extension {extension(path)}')

def to_tensor(path: str) -> Tensor:
    """Convert the source into a tensor.
    """
    ext = extension(path)
    if ext in c.NUMPY_COMPATIBLE_EXTENSIONS:
        return torch.from_numpy(np.load(path))
    if ext in c.IMAGE_EXTENSIONS:
        return ToPILImage()(to_image(path))
    return torch.load(path)

def unpickle(path:str) -> object:
    """Open a pickled file and return the object"""
    with open(path, 'rb') as file:
        return pickle.load(file)

### Savers ###

def from_numpy(array: NumpyArray, path: str) -> None:
    """Save numpy array to path.
    """
    np.savetxt(path, array)

def from_image(image: Image, path: str) -> None:
    """Save image to path.
    """
    image.save(path)

def from_dataframe(df: pd.DataFrame, path: str) -> None:
    """Save dataframe to path.
    """
    df.to_csv(path)

def from_tensor(tensor: Tensor, path: str) -> None:
    """Save tensor to path.
    """
    torch.save(tensor, path)

def pickle_from(obj: object, path: str) -> None:
    """Save object to path.
    """
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

### Path ###

def extension(path: str) -> str:
    """Return the extension of the file"""
    return os.path.splitext(path)[1]

def filename(path: str) -> str:
    """Return the filename of the file"""
    return os.path.splitext(os.path.basename(path))[0]
