from __future__ import annotations
from typing import Any
from torch import Tensor
import numpy as np
from .entity import Entity
from ..types_alias import Source3DVector
import torch


class Vector(Entity):
    """Define a vector in 3D
    Uses a 3D torch tensor internally to represent the vector.
    It is stored in SI unit: meters, but supports quick conversion from and to millimeters.
    """

    def __init__(
        self,
        source: Source3DVector | Vector,
        name: str = "vector",
        reference_frame: str | None = None,
        extra: Any | None = None,
        millimeters: bool = False,
    ) -> None:
        """Create a new entity"""
        if isinstance(source, Vector):
            source = source.tensor.clone()
        if millimeters:
            source = torch.as_tensor(source) / 1000
        super().__init__(source, name, reference_frame, extra)

    ######################
    ###   Components   ###
    ######################

    @property
    def x(self) -> float:
        """Return the x component of the translation"""
        return self.tensor[0].item()

    @x.setter
    def x(self, x: float) -> None:
        """Set the x component of the translation"""
        self.tensor[0] = x

    @property
    def y(self) -> float:
        """Return the y component of the translation"""
        return self.tensor[1].item()

    @y.setter
    def y(self, y: float) -> None:
        """Set the y component of the translation"""
        self.tensor[1] = y

    @property
    def z(self) -> float:
        """Return the z component of the translation"""
        return self.tensor[2].item()

    @z.setter
    def z(self, z: float) -> None:
        """Set the z component of the translation"""
        self.tensor[2] = z

    def norm(self) -> float:
        """Return the norm of the vector"""
        return float(np.linalg.norm(self.numpy()))

    def unit(self) -> Vector:
        """Return the unit vector
        """
        return Vector(
            self.tensor / self.norm(),
            name=f"unit {self.name}",
            reference_frame=self.reference_frame
        )

    ######################
    ###   Operations   ###
    ######################

    def __dot__(self, other: Vector) -> float:
        """Dot product"""
        return self.tensor.dot(other.tensor).item()

    def __mul__(self, other: Vector) -> float:
        """Multiply two poses together"""
        return self.__dot__(other)

    def __rmul__(self, other: Vector) -> float:
        """Multiply two poses together"""
        return other.__dot__(self)

    def cross(self, other: Vector) -> Vector:
        """Cross product"""
        return Vector(self.tensor.cross(other.tensor))

    def __add__(self, other: Vector) -> Vector:
        """Add two poses together"""
        return Vector(
            self.tensor + other.tensor,
            name=f"{self.name} + {other.name}",
            reference_frame=self.reference_frame,
            extra=(self, other)
        )

    def __sub__(self, other: Vector) -> Vector:
        """Subtract two poses together"""
        return Vector(
            self.tensor - other.tensor,
            name=f"{self.name} - {other.name}",
            reference_frame=self.reference_frame,
            extra=(self, other)
        )

    def __rsub__(self, other: Vector) -> Vector:
        """Subtract two poses together"""
        return Vector(
            other.tensor - self.tensor,
            name=f"{other.name} - {self.name}",
            reference_frame=self.reference_frame,
            extra=(other, self)
        )

    ########################
    ###   Construction   ###
    ########################

    @classmethod
    def from_xyz(
        cls,
        x: float = 0,
        y: float = 0,
        z: float = 0,
        name: str = "vector",
        reference_frame: str | None = None
    ) -> Vector:
        """Create a vector from x, y, z components"""
        return Vector(Tensor([x, y, z]), name=name, reference_frame=reference_frame)

    @classmethod
    def identity(
        cls,
        name: str = "vector",
        reference_frame: str | None = None,
    ) -> Vector:
        """Return the identity vector"""
        return cls(
            torch.zeros(3),
            name=name,
            reference_frame=reference_frame,
        )

    @classmethod
    def x0(
        cls,
        name: str = "vector",
        reference_frame: str | None = None,
    ) -> Vector:
        """Return the x0 vector"""
        return cls(
            [1, 0, 0],
            name=name,
            reference_frame=reference_frame,
        )

    @classmethod
    def y0(
        cls,
        name: str = "vector",
        reference_frame: str | None = None,
    ) -> Vector:
        """Return the y0 vector"""
        return cls(
            [0, 1, 0],
            name=name,
            reference_frame=reference_frame,
        )

    @classmethod
    def z0(
        cls,
        name: str = "vector",
        reference_frame: str | None = None,
    ) -> Vector:
        """Return the z0 vector"""
        return cls(
            [0, 0, 1],
            name=name,
            reference_frame=reference_frame,
        )

    ############################
    ###   Units conversion   ###
    ############################

    @classmethod
    def from_millimeters(cls, millimeters: Source3DVector) -> Vector:
        """Create a vector from millimeters"""
        return Vector(Tensor(millimeters) / 1000)

    @property
    def millimeters(self) -> Tensor:
        """Return the vector in millimeters"""
        return self.tensor * 1000

    @millimeters.setter
    def millimeters(self, millimeters: Source3DVector) -> None:
        """Update the value from a value in millimeters"""
        self.tensor = Tensor(millimeters) / 1000
