from __future__ import annotations
from typing import Any, Dict, Tuple
from torch import Tensor
import pickle
from copy import deepcopy
import numpy as np
import PIL.Image
import os
import torch
import torchvision.transforms as t
from PIL.Image import Image
import matplotlib.pyplot as plt

def to_tensor(
        image: Image,
        transforms: t.Compose | None = None
    ) -> Tensor | None:
    """Convert an image into a tensor.
    """
    if transforms is not None:
        transformed = transforms(image)
        if isinstance(transformed, Tensor):
            return transformed
    return t.ToTensor()(image)

def from_tensor(tensor: Tensor) -> Image:
    """Convert a tensor into an image.
    TODO: add denormalization
    """
    return t.ToPILImage()(tensor)

def load(path: str) -> Image | None:
    """Load the item as an image.
    """
    return PIL.Image.open(path)


def save(image: Image | Tensor, path: str) -> None:
    """Save the item as an image.
    """
    if isinstance(image, Tensor):
        image = from_tensor(image)
    image.save(path)

def show(image: Image | Tensor) -> None:
    """Show the item as an image.
    """
    if isinstance(image, Tensor):
        image = from_tensor(image)
    image.show()
    
def plot(
        image: Image | Tensor,
        title: str = "Image",
        figsize: Tuple[int, int] = (10, 10)
    ):
        """Plot the image"""
        if isinstance(image, Tensor):
            image = from_tensor(image)
        plt.figure(figsize=figsize)  # type: ignore
        if title is not None:
            plt.title(title)  # type: ignore
        plt.imshow(image)  # type: ignore
        plt.show()  # type: ignore