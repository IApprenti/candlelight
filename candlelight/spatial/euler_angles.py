from __future__ import annotations
from typing import Any
from torch import Tensor
import numpy as np

from .. import constants
from .entity import Entity
from ..types_alias import Source3DVector
import torch
from .vector import Vector


class EulerAngles(Entity):
    """Define a vector of euler angles in 3D
    Uses a 3D torch tensor internally to represent the vector.
    """

    # The convention used to represent the angles
    # We use the same conventions as in the scipy.spatial.transform.Rotation class
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    # Lower case for extrinsic rotations, upper case for intrinsic rotations
    # The default convention is a roll-pitch-yaw around a fixed frame: xyz extrinsic
    RPY_CONVENTION = constants.RPY_CONVENTION

    def __init__(
        self,
        source: Source3DVector | EulerAngles,
        name: str = "euler angles",
        reference_frame: str | None = None,
        extra: Any | None = None,
        convention: str = RPY_CONVENTION,
        degrees: bool = False,
    ) -> None:
        """Create a new entity"""
        if isinstance(source, EulerAngles):
            source = source.tensor.clone()
        source = torch.as_tensor(source)
        if degrees:
            source[3:6] = torch.deg2rad(source[3:6])
        super().__init__(source, name, reference_frame, extra)
        self.convention: str = convention

    ######################
    ###   Components   ###
    ######################

    @property
    def a(self) -> float:
        """Return the 1st component of the vector"""
        return self.tensor[0].item()

    @a.setter
    def a(self, a: float) -> None:
        """Set the 1st component of the vector"""
        self.tensor[0] = a

    @property
    def b(self) -> float:
        """Return the 2nd component of the vector"""
        return self.tensor[1].item()

    @b.setter
    def b(self, b: float) -> None:
        """Set the 2nd component of the vector"""
        self.tensor[1] = b

    @property
    def c(self) -> float:
        """Return the 3rd component of the vector"""
        return self.tensor[2].item()

    @c.setter
    def c(self, c: float) -> None:
        """Set the 3rd component of the vector"""
        self.tensor[2] = c

    ######################
    ###   Operations   ###
    ######################

    def __add__(self, other: EulerAngles) -> EulerAngles:
        """Add two vectors"""
        return EulerAngles(
            self.tensor + other.tensor,
            name=self.name,
            reference_frame=self.reference_frame,
            convention=self.convention,
        )

    ########################
    ###   Construction   ###
    ########################

    @classmethod
    def from_abc(
        cls,
        a: float = 0,
        b: float = 0,
        c: float = 0,
        name: str = "angles",
        reference_frame: str | None = None
    ) -> EulerAngles:
        """Create a angles from components"""
        return EulerAngles(Tensor([a, b, c]), name=name, reference_frame=reference_frame)

    @classmethod
    def identity(
        cls,
        name: str = "angles",
        reference_frame: str | None = None,
    ) -> EulerAngles:
        """Return the identity angles"""
        return cls(
            torch.zeros(3),
            name=name,
            reference_frame=reference_frame,
        )

    @classmethod
    def a0(
        cls,
        name: str = "angles",
        reference_frame: str | None = None,
    ) -> EulerAngles:
        """Return the first 0 vector"""
        return cls(
            [1, 0, 0],
            name=name,
            reference_frame=reference_frame,
        )

    @classmethod
    def b0(
        cls,
        name: str = "angles",
        reference_frame: str | None = None,
    ) -> EulerAngles:
        """Return the second 0 vector"""
        return cls(
            [0, 1, 0],
            name=name,
            reference_frame=reference_frame,
        )

    @classmethod
    def c0(
        cls,
        name: str = "angles",
        reference_frame: str | None = None,
    ) -> EulerAngles:
        """Return the third 0 vector"""
        return cls(
            [0, 0, 1],
            name=name,
            reference_frame=reference_frame,
        )

    ##########################
    ###   Roll-Pitch-Yaw   ###
    ##########################

    @property
    def roll(self) -> float:
        """Return the roll angle"""
        if self.convention == EulerAngles.RPY_CONVENTION:
            return self.a
        raise NotImplementedError(
            f"No roll angle for convention {self.convention}")

    @roll.setter
    def roll(self, roll: float) -> None:
        """Set the roll angle"""
        if self.convention == EulerAngles.RPY_CONVENTION:
            self.a = roll
        else:
            raise NotImplementedError(
                f"No roll angle for convention {self.convention}")

    @property
    def pitch(self) -> float:
        """Return the pitch angle"""
        if self.convention == EulerAngles.RPY_CONVENTION:
            return self.b
        raise NotImplementedError(
            f"No pitch angle for convention {self.convention}")

    @pitch.setter
    def pitch(self, pitch: float) -> None:
        """Set the pitch angle"""
        if self.convention == EulerAngles.RPY_CONVENTION:
            self.b = pitch
        else:
            raise NotImplementedError(
                f"No pitch angle for convention {self.convention}")

    @property
    def yaw(self) -> float:
        """Return the yaw angle"""
        if self.convention == EulerAngles.RPY_CONVENTION:
            return self.c
        raise NotImplementedError(
            f"No yaw angle for convention {self.convention}")

    @yaw.setter
    def yaw(self, yaw: float) -> None:
        """Set the yaw angle"""
        if self.convention == EulerAngles.RPY_CONVENTION:
            self.c = yaw
        else:
            raise NotImplementedError(
                f"No yaw angle for convention {self.convention}")

    ############################
    ###   Units conversion   ###
    ############################

    @classmethod
    def from_degrees(
        cls,
        degrees: Source3DVector,
        name="angles",
        reference_frame: str | None = None,
        convention: str = RPY_CONVENTION,
    ) -> EulerAngles:
        """Create a angles from degrees"""
        return EulerAngles(
            torch.deg2rad(Tensor(degrees)),
            name=name,
            reference_frame=reference_frame,
            convention=convention,
        )

    @property
    def degrees(self) -> Tensor:
        """Return the vector in millimeters"""
        return torch.rad2deg(self.tensor)

    @degrees.setter
    def degrees(self, degrees: Source3DVector) -> None:
        """Update the value from a value in millimeters"""
        self.tensor = torch.deg2rad(Tensor(degrees))
