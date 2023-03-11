"""Manage images as torch tensors"""

from __future__ import annotations
import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import torch
import torchvision
from torch import Tensor
from copy import deepcopy

from ..types_alias import NumpyArray

from .. import helpers


class Image:
    """An images class that store one image in memory in the form of a tensor"""

    def __init__(
        self,
        tensor: Tensor,
        conform: bool | None = None,
        labels: List[str] | None = None,
        name: str | None = None,
        timestamp: float | None = None,
    ):
        self.tensor: Tensor = tensor
        self.conform: bool | None = conform
        self.labels: List[str] | None = labels
        self.name: str | None = name
        self.timestamp: float = timestamp if timestamp is not None else helpers.current_timestamp()

    def __str__(self) -> str:
        """String representation"""
        return f"""Image with:
name = {self.name}
timestamp = {helpers.timestamp_to_prefix(self.timestamp)}
conform = {self.conform}
labels = {self.labels}
tensor.shape = {self.tensor.shape if self.tensor is not None else None}
"""

    @property
    def legend(self) -> str:
        """Write a human readable legend"""
        return f"{self.name} - {self.conform_string} - {self.labels}"

    @property
    def conform_string(self) -> str:
        """Return a string saying if the image is conform"""
        if self.conform is None:
            return "unknown conformity"
        return "conform" if self.conform else "anomalous"

    ###################
    ###   Helpers   ###
    ###################

    @staticmethod
    def convert_image_to_tensor(pil_image: PILImage.Image):
        """Convert a PIL or numpy image to a tensor"""
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )
        # save_image(transform(pil_image), "debug_convert_to_tensor.png")
        return transform(pil_image)

    @staticmethod
    def convert_path_to_pill_image(path: str) -> PILImage.Image:
        """Convert a path to a PIL image
        """
        pil = PILImage.open(path)
        return pil

    def copy(self) -> Image:
        """Return a copy of the image
        """
        return deepcopy(self)

    #########################
    ###   Get the image   ###
    #########################

    @staticmethod
    def image_tensor_from_batch_tensor(batch: Tensor, index: int = 0) -> Tensor:
        """Extract one image from a batch at index
        Store it as self.tensor

        Typical PyTorch images batch tensor: torch.Size([N, C, H, W])
        N — batch size (number of images per batch)
        C — number of channels (usually uses 3 channels for RGB)
        H — height of the image
        W — width of the image

        :param batch: A tensor representing a batch of images
        :param index: The index of the desired image (default to 0)
        :return: The tensor of the image at index
        """
        return batch[index]

    def get_image_tensor_from_batch_tensor(self, batch: Tensor, index: int = 0) -> Tensor:
        """Extract one image from a batch at index
        Store it as self.tensor

        Typical PyTorch images batch tensor: torch.Size([N, C, H, W])
        N — batch size (number of images per batch)
        C — number of channels (usually uses 3 channels for RGB)
        H — height of the image
        W — width of the image

        :param batch: A tensor representing a batch of images
        :param index: The index of the desired image (default to 0)
        :return: The tensor of the image at index
        """
        self.tensor: Tensor = self.image_tensor_from_batch_tensor(
            batch=batch,
            index=index
        )
        return self.tensor

    @staticmethod
    def tensor_from_image_file_path(path: str) -> Tensor:
        """Return the tensor of an image from a file path
        """
        pil_image: PILImage.Image = Image.convert_path_to_pill_image(path)
        tensor: Tensor = Image.convert_image_to_tensor(pil_image)
        return tensor

    def get_tensor_from_image_file_path(self, path: str) -> Tensor:
        """Get the tensor of an image from a file path
        """
        self.tensor = Image.tensor_from_image_file_path(path)
        return self.tensor

    ####################################
    ###   Alternative constructors   ###
    ####################################

    @classmethod
    def from_pil_image(
        cls,
        pil_image: PILImage.Image,
        conform: bool | None = None,
        labels: List[str] | None = None,
        name: str | None = None,
    ) -> Image:
        """Create an image object from a PIL image"""
        return cls(
            tensor=Image.convert_image_to_tensor(pil_image),
            conform=conform,
            labels=labels,
            name=name,
        )

    @classmethod
    def from_batch_tensor(
        cls,
        batch: Tensor,
        index: int = 0,
        conform: bool | None = None,
        labels: List[str] | None = None,
        name: str | None = None,
    ):
        """Create an image object from a batch of images tensor or from a single image tensor"""
        is_batch = len(batch.shape) == 4
        is_image = len(batch.shape) == 3
        if not is_batch and not is_image:
            raise ValueError(
                f"Batch should be a 4D tensor or a 3D tensor, got {batch.shape}")
        if is_image:
            return cls(
                tensor=batch,
                conform=conform,
                labels=labels,
                name=name,
            )
        else:
            return cls(
                tensor=batch[index],
                conform=conform,
                labels=labels,
                name=name,
            )

    @classmethod
    def from_file_path(
        cls,
        path: str,
        conform: bool | None = None,
        labels: List[str] | None = None,
        name: str | None = None,
    ) -> Image:
        """Create an image object for a file path"""
        return cls(
            tensor=Image.tensor_from_image_file_path(path=path),
            conform=conform,
            labels=labels,
            name=name,
        )

    @classmethod
    def random(cls):
        """Create a dummy image"""
        return cls(
            tensor=torch.rand(3, 256, 256)
        )

    @classmethod
    def dummy(
        cls,
        path: str,
        conform: bool = True,
        name: str | None = None,
        labels: List[str] | None = None,
    ) -> Image:
        """Create a dummy image"""
        if name is None:
            name = "dummy_image"
        # Create the image
        return cls.from_file_path(
            path=path,
            conform=conform,
            labels=labels,
            name=name
        )

    @classmethod
    def random_noise(
        cls,
        width: int = 300,
        height: int = 400,
        name: str | None = None,
        conform: bool = True,
        labels: List[str] | None = None,
    ):
        """Create an image filled with noise"""
        # Args
        tensor = torch.rand(3, width, height)
        # Name
        if name is None:
            name = "dummy_image"
        return cls(
            tensor=tensor,
            conform=conform,
            labels=labels
        )

    ######################
    ###   Properties   ###
    ######################

    @property
    def numpy_rgb(self) -> NumpyArray:
        """Export in Numpy (w,h,3) format"""
        return self.tensor.permute(1, 2, 0).numpy()

    @property
    def batch(self) -> Tensor:
        """Return a batch tensor with only the image"""
        return self.tensor.unsqueeze(0)

    @property
    def number_of_channels(self) -> int:
        """Gives the number of channels of the image"""
        return self.tensor.shape[0]

    @property
    def height(self) -> int:
        """Gives the height of the image"""
        return self.tensor.shape[1]

    @property
    def width(self) -> int:
        """Gives the width of the image"""
        return self.tensor.shape[2]

    @property
    def dimensions(self) -> Tuple[int, int]:
        """Gives the dimensions of the image as a tuple (height, width)"""
        return self.height, self.width

    @property
    def red(self) -> Tensor | None:
        """Gives the red channel of the image"""
        if self.number_of_channels > 2:
            return self.tensor[0]
        return None

    @property
    def green(self) -> Tensor | None:
        """Gives the green channel of the image"""
        if self.number_of_channels > 2:
            return self.tensor[1]
        return None

    @property
    def blue(self) -> Tensor | None:
        """Gives the blue channel of the image"""
        if self.number_of_channels > 2:
            return self.tensor[2]
        return None

    @property
    def alpha(self) -> Tensor | None:
        """Gives the alpha channel of the image"""
        if self.number_of_channels > 3:
            return self.tensor[3]
        return None

    @property
    def grey(self) -> Tensor | None:
        """Gives the grey channel of the image"""
        if self.number_of_channels == 1:
            return self.tensor[0]
        elif self.number_of_channels == 3:
            return self.tensor.mean(0)
        elif self.number_of_channels == 4:
            return self.tensor[:3].mean(0)
        return None

    @property
    def rvb(self) -> Tensor | None:
        """Return the image with 3 channels: r, v, b"""
        if self.number_of_channels == 1:
            grey: Tensor = self.tensor[0]
            return torch.stack([grey, grey, grey], dim=0)
        elif self.number_of_channels == 3:
            return self.tensor
        elif self.number_of_channels == 4:
            return self.tensor[:3]
        return None

    @property
    def rvba(self) -> Tensor | None:
        """Return the image with 4 channels: r, v, b, a"""
        if self.number_of_channels == 1:
            tensor: Tensor = torch.cat(
                [
                    self.tensor[0],
                    self.tensor[0],
                    self.tensor[0],
                    torch.ones([1, self.height, self.width])
                ],
                dim=0
            )
            return tensor
        elif self.number_of_channels == 3:
            return torch.cat((self.tensor, torch.ones((1, self.height, self.width))), dim=0)
        elif self.number_of_channels == 4:
            return self.tensor
        return None

    @property
    def channels_at_the_end(self) -> Tensor:
        """Return the image with channels at the end (format used by other libraries such as MatPlotLib or TensorFlow)"""
        return self.tensor.permute(1, 2, 0)

    @property
    def pil(self) -> PILImage.Image:
        """Returns a PIL image"""
        image: PILImage.Image = torchvision.transforms.ToPILImage()(self.tensor)
        return image

    @property
    def label(self) -> str | None:
        """Returns the first label of the image"""
        if self.labels is None:
            return None
        return self.labels[0]

    @label.setter
    def label(self, label: str):
        """Sets the first label of the image"""
        if self.labels is None:
            self.labels = [label]
        else:
            self.labels[0] = label

    ######################
    ###   Save image   ###
    ######################

    def save(
        self,
        path: str = "./logs",
        name: str | None = None,
        extension: str = "png",
        use_timestamp: bool = True,
    ):
        """Save the image to a file
        use png by default
        """
        # Create the directory if it does not exist
        helpers.create_directory(path)
        # Create the filename
        name = name if name is not None else self.name
        if name is None:
            name = "image"
        if use_timestamp:
            prefix = helpers.timestamp_to_prefix(self.timestamp)
            name = f"{prefix} - {name}"
        filename: str = f"{name}.{extension}"
        # Get the path
        path = os.path.join(path, filename)
        # Save the image
        self.save_image(path)

    def save_image(self, path: str):
        """Save an image on the disk
        Use an extension like .png
        """
        torchvision.utils.save_image(  # type: ignore
            tensor=self.tensor,
            fp=path
        )

    def save_tensor(self, path: str):
        """Save a tensor on the disk"""
        torch.save(  # type: ignore
            obj=self.tensor,
            f=path
        )

    #########################
    ###   Display image   ###
    #########################

    def plot(
        self,
        title: str = "Image",
        figsize: Tuple[int, int] = (10, 10)
    ):
        """Plot the image"""
        plt.figure(figsize=figsize)  # type: ignore
        if title is not None:
            plt.title(title)  # type: ignore
        plt.imshow(self.channels_at_the_end)  # type: ignore
        plt.show()  # type: ignore

    ##############################
    ###   Image manipulation   ###
    ##############################

    def resize(self, size: Tuple[int, int]):
        """Resize the image"""
        self.tensor = torchvision.transforms.Resize(size=size)(self.tensor)

    def add_dummy_anomaly(self):
        """Add a dummy anomaly to the image"""
        self.tensor[0, 10:20, 0] = 0.5
