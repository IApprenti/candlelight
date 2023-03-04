from __future__ import annotations
from typing import Any, List
import torch
from torch import Tensor
import numpy as np
from ..types_alias import Numpy3DVector, Source3DPose, Source3DVector, Source3DRotation
from .vector import Vector
from .entity import Entity
from ..constants import index_names
from .. import constants as c


class Rotation(Entity):
    """Define a frame orientation in 3D
    Uses a 3x3 rotation matrix internally as a tensor to represent the rotation
    """

    def __init__(
        self,
        source: Source3DRotation | Rotation,
        name: str = "pose",
        reference_frame: str | None = None,
        extra: Any | None = None,
    ) -> None:
        """Create a new pose"""
        if isinstance(source, Rotation):
            source = source.tensor.clone()
        super().__init__(source, name, reference_frame, extra)

    ######################
    ###   Components   ###
    ######################

    def get_frame_axis(self, index: int):
        """Get an axis of the rotation frame"""
        return Vector(
            source=self.tensor[0:2, index],
            name=f"{self.name}'s {index_names[index]} axis",
            reference_frame=self.reference_frame,
        )

    def set_frame_axis(self, vector: Vector, index: int):
        """Set an axis of the rotation frame"""
        if not isinstance(vector, Vector):
            vector = Vector(vector)
        self.tensor[0:2, index] = vector.tensor

    @property
    def x_axis(self) -> Vector:
        """Return the x axis of the rotation frame"""
        return self.get_frame_axis(c.X_INDEX)

    @x_axis.setter
    def x_axis(self, x: Vector) -> None:
        """Set the x axis of the rotation frame"""
        self.set_frame_axis(x, c.X_INDEX)

    @property
    def y_axis(self) -> Vector:
        """Return the y axis of the rotation frame"""
        return self.get_frame_axis(c.Y_INDEX)

    @y_axis.setter
    def y_axis(self, y: Vector) -> None:
        """Set the y axis of the rotation frame"""
        self.set_frame_axis(y, c.Y_INDEX)

    @property
    def z_axis(self) -> Vector:
        """Return the z axis of the rotation frame"""
        return self.get_frame_axis(c.Z_INDEX)

    @z_axis.setter
    def z_axis(self, z: Vector) -> None:
        """Set the z axis of the rotation frame"""
        self.set_frame_axis(z, c.Z_INDEX)

    ######################
    ###   Operations   ###
    ######################

    def __dot__(self, other: Rotation) -> Rotation:
        """Multiply two poses together"""
        return Rotation(self.tensor @ other.tensor)

    def __mul__(self, other: Rotation) -> Rotation:
        """Multiply two poses together"""
        return self.__dot__(other)

    def __rmul__(self, other: Rotation) -> Rotation:
        """Multiply two poses together"""
        return other.__dot__(self)

    ########################
    ###   Constructors   ###
    ########################

    @classmethod
    def identity(
        cls,
        name: str = "vector",
        reference_frame: str | None = None,
    ) -> Rotation:
        """Return the identity rotation"""
        return cls(
            torch.eye(3),
            name=name,
            reference_frame=reference_frame,
        )
