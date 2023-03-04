from __future__ import annotations
from typing import Any
from torch import Tensor
import numpy as np

from .. import constants
from .entity import Entity
from ..types_alias import Source3DVector, SourceTwist
import torch
from .vector import Vector


class Twist(Entity):
    """A Twist is a common representation of a pose in 3D
    Uses a 3D torch tensor internally to represent the vector.
    If no extra is provided, anything after index 5 is considered to be extra.
    """

    # The convention used to represent the angles
    # We use the same conventions as in the scipy.spatial.transform.Rotation class
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    # Lower case for extrinsic rotations, upper case for intrinsic rotations
    # The default convention is a roll-pitch-yaw around a fixed frame: xyz extrinsic
    RPY_CONVENTION = constants.RPY_CONVENTION

    def __init__(
        self,
        source: SourceTwist | Twist,
        name: str = "euler angles",
        reference_frame: str | None = None,
        extra: Any | None = None,
        convention: str = RPY_CONVENTION,
        degrees: bool = False,
        millimeters: bool = False,
    ) -> None:
        """Create a new entity"""
        if isinstance(source, Twist):
            if extra is None:
                extra = source.extra
            source = source.tensor.clone()
        if len(source) == 7 and extra is None:
            extra = source[6]
        if len(source) > 7 and extra is None:
            extra = source[6:]
        source = torch.as_tensor(source[:6])
        if millimeters:
            source[0:3] = source[0:3] / 1000
        if degrees:
            source[3:6] = torch.deg2rad(source[3:6])
        super().__init__(source, name, reference_frame, extra)
        self.convention: str = convention

    ############################
    ###   Angle Components   ###
    ############################

    @property
    def a(self) -> float:
        """Return the 1st component of the angles vector"""
        return self.tensor[3].item()

    @a.setter
    def a(self, a: float) -> None:
        """Set the 1st component of the angles vector"""
        self.tensor[3] = a

    @property
    def b(self) -> float:
        """Return the 2nd component of the angles vector"""
        return self.tensor[4].item()

    @b.setter
    def b(self, b: float) -> None:
        """Set the 2nd component of the angles vector"""
        self.tensor[4] = b

    @property
    def c(self) -> float:
        """Return the 3rd component of the angles vector"""
        return self.tensor[5].item()

    @c.setter
    def c(self, c: float) -> None:
        """Set the 3rd component of the angles vector"""
        self.tensor[5] = c

    ######################
    ###   Operations   ###
    ######################

    def __add__(self, other: Twist) -> Twist:
        """Add two vectors
        WARNING: this is not a geometric addition yet, it is a vector addition
        """
        extra = None
        try:
            # TODO: make something cleaner
            extra = self.extra + other.extra if self.extra and other.extra else None
        except:
            pass
        if extra is None and self.extra is not None:
            extra = self.extra
        elif extra is None and other.extra is not None:
            extra = other.extra
        # TODO: check out the angles for being between -pi and pi or 0 and 2pi
        return Twist(
            self.tensor + other.tensor,
            name=self.name,
            reference_frame=self.reference_frame,
            convention=self.convention,
            extra=extra,
        )

    def __sub__(self, other: Twist) -> Twist:
        """Subtract two vectors
        WARNING: this is not a geometric subtraction yet, it is a vector subtraction
        """
        extra = None
        try:
            # TODO: make something cleaner
            extra = self.extra - other.extra if self.extra and other.extra else None
        except:
            pass
        if extra is None and self.extra is not None:
            extra = self.extra
        elif extra is None and other.extra is not None:
            extra = other.extra
        # TODO: check out the angles for being between -pi and pi or 0 and 2pi
        return Twist(
            self.tensor - other.tensor,
            name=self.name,
            reference_frame=self.reference_frame,
            convention=self.convention,
            extra=extra,
        )

    def __rsub__(self, other: Twist) -> Twist:
        """Subtract two vectors
        WARNING: this is not a geometric subtraction yet, it is a vector subtraction
        """
        extra = None
        try:
            # TODO: make something cleaner
            extra = other.extra - self.extra if self.extra and other.extra else None
        except:
            pass
        if extra is None and self.extra is not None:
            extra = self.extra
        elif extra is None and other.extra is not None:
            extra = other.extra
        # TODO: check out the angles for being between -pi and pi or 0 and 2pi
        return Twist(
            other.tensor - self.tensor,
            name=self.name,
            reference_frame=self.reference_frame,
            convention=self.convention,
            extra=extra,
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
    ) -> Twist:
        """Create a angles from components"""
        return Twist(Tensor([a, b, c]), name=name, reference_frame=reference_frame)

    @classmethod
    def identity(
        cls,
        name: str = "angles",
        reference_frame: str | None = None,
    ) -> Twist:
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
    ) -> Twist:
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
    ) -> Twist:
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
    ) -> Twist:
        """Return the third 0 vector"""
        return cls(
            [0, 0, 1],
            name=name,
            reference_frame=reference_frame,
        )

    ##############################
    ###   Translational part   ###
    ##############################

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

    ##########################
    ###   Roll-Pitch-Yaw   ###
    ##########################

    @property
    def roll(self) -> float:
        """Return the roll angle"""
        if self.convention == Twist.RPY_CONVENTION:
            return self.a
        raise NotImplementedError(
            f"No roll angle for convention {self.convention}")

    @roll.setter
    def roll(self, roll: float) -> None:
        """Set the roll angle"""
        if self.convention == Twist.RPY_CONVENTION:
            self.a = roll
        else:
            raise NotImplementedError(
                f"No roll angle for convention {self.convention}")

    @property
    def pitch(self) -> float:
        """Return the pitch angle"""
        if self.convention == Twist.RPY_CONVENTION:
            return self.b
        raise NotImplementedError(
            f"No pitch angle for convention {self.convention}")

    @pitch.setter
    def pitch(self, pitch: float) -> None:
        """Set the pitch angle"""
        if self.convention == Twist.RPY_CONVENTION:
            self.b = pitch
        else:
            raise NotImplementedError(
                f"No pitch angle for convention {self.convention}")

    @property
    def yaw(self) -> float:
        """Return the yaw angle"""
        if self.convention == Twist.RPY_CONVENTION:
            return self.c
        raise NotImplementedError(
            f"No yaw angle for convention {self.convention}")

    @yaw.setter
    def yaw(self, yaw: float) -> None:
        """Set the yaw angle"""
        if self.convention == Twist.RPY_CONVENTION:
            self.c = yaw
        else:
            raise NotImplementedError(
                f"No yaw angle for convention {self.convention}")

    ############################
    ###   Units conversion   ###
    ############################

    @classmethod
    def from_mm_degrees(
        cls,
        millimeters: Source3DVector,
        degrees: Source3DVector,
        name="angles",
        reference_frame: str | None = None,
        convention: str = RPY_CONVENTION,
    ) -> Twist:
        """Create a angles from degrees"""
        translation = Tensor(millimeters)
        rotation = torch.deg2rad(Tensor(degrees))
        return Twist(
            torch.concat((translation, rotation)),
            name=name,
            reference_frame=reference_frame,
            convention=convention,
        )

    @property
    def mm_degrees(self) -> Tensor:
        """Return the twist as a tensor in millimeters and degrees"""
        millimeters = self.tensor[0:3] * 1000
        degrees = torch.rad2deg(self.tensor[3:5])
        return torch.concat((millimeters, degrees))

    @property
    def m_degrees(self) -> Tensor:
        """Return the twist as a tensor in meters and degrees"""
        meters = self.tensor[0:3]
        degrees = torch.rad2deg(self.tensor[3:5])
        return torch.concat((meters, degrees))
