"""Module handling batch processing of images."""

from __future__ import annotations
import os
from typing import List, Tuple
import torch
from torch import Tensor
from .image import Image


class Batch:
    """Manage batches of images and their labels"""

    def __init__(
        self,
        tensor: Tensor,
        labels: List[str] | None = None,
    ):
        self.tensor: Tensor = tensor
        self.labels: List[str] = labels

    ###################
    ###   Getters   ###
    ###################

    def image(self, index: int) -> Image:
        """Return an image object from the batch"""
        return Image.from_batch_tensor(
            batch=self.tensor,
            index=index,
            labels=[self.labels[index]]
        )

    @property
    def images(self) -> List[Image]:
        """Return a list of images from the batch"""
        return [self.image(index) for index in range(len(self))]

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Return the shape of the batch"""
        return self.tensor.shape

    @property
    def length(self) -> int:
        """Return the length of the batch"""
        return len(self)

    ##################
    ###   Setters  ###
    ##################

    def add_image(self, image: Image) -> None:
        """Add an image to the batch"""
        self.tensor = torch.cat([self.tensor, image.batch])
        self.labels.append(image.labels)

    def remove_image(self, index: int) -> None:
        """Remove an image from the batch"""
        self.tensor = torch.cat([self.tensor[:index], self.tensor[index+1:]])
        self.labels.pop(index)

    def replace_image(self, index: int, image: Image) -> None:
        """Replace an image in the batch"""
        self.tensor[index] = image.tensor
        self.labels[index] = image.labels

    ####################################
    ###   Alternative constructors   ###
    ####################################

    @classmethod
    def from_image(cls, image: Image) -> Batch:
        """Create a batch from a single image"""
        return cls(
            tensor=image.batch,
            labels=[image.labels],
        )

    @classmethod
    def from_images(cls, images: List[Image]) -> Batch:
        """Create a batch from a list of images"""
        return cls(
            tensor=torch.stack([image.tensor for image in images]),
            labels=[image.labels for image in images],
        )

    @classmethod
    def from_image_paths(cls, image_paths: List[str]) -> Batch:
        """Create a batch from a list of image paths"""
        images = []
        for path in image_paths:
            directory_name = os.path.basename(os.path.dirname(path))
            images.append(Image.from_file_path(
                path=path, labels=[directory_name]))
        return cls.from_images(images=images)
