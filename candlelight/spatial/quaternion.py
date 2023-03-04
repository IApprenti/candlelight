from __future__ import annotations
from typing import Tuple, Any
from torch import Tensor
import numpy as np
from .entity import Entity
from ..types_alias import SourceQuaternion
from .. import constants as c


class Quaternion(Entity):
    """Define a vector in 3D
    Uses a 3D torch tensor internally to represent the vector.
    It is considered to be in SI, so: meters.
    """

    def __init__(
        self,
        source: SourceQuaternion | Quaternion,
        name: str = "quaternion",
        reference_frame: str | None = None,
        extra: Any | None = None,
    ) -> None:
        """Create a new entity"""
        if isinstance(source, Quaternion):
            source = source.tensor.clone()
        super().__init__(source, name, reference_frame, extra)

    ######################
    ###   Components   ###
    ######################

    @property
    def x(self) -> float:
        """Return the x component of the translation"""
        return self.tensor[c.X_INDEX].item()

    @x.setter
    def x(self, x: float) -> None:
        """Set the x component of the translation"""
        self.tensor[c.X_INDEX] = x

    @property
    def y(self) -> float:
        """Return the y component of the translation"""
        return self.tensor[c.Y_INDEX].item()

    @y.setter
    def y(self, y: float) -> None:
        """Set the y component of the translation"""
        self.tensor[c.Y_INDEX] = y

    @property
    def z(self) -> float:
        """Return the z component of the translation"""
        return self.tensor[c.Z_INDEX].item()

    @z.setter
    def z(self, z: float) -> None:
        """Set the z component of the translation"""
        self.tensor[c.Z_INDEX] = z

    @property
    def w(self) -> float:
        """Return the w component of the translation"""
        return self.tensor[c.W_INDEX].item()

    @w.setter
    def w(self, w: float) -> None:
        """Set the w component of the translation"""
        self.tensor[c.W_INDEX] = w

    ####################
    ###   Calculus   ###
    ####################

    def modulus(self) -> float:
        """Return the norm of the quaternion"""
        return float(np.linalg.norm(self.numpy()))

    def unit(self) -> Quaternion:
        """Return the unit quaternion
        Note that rotations usually use unit quaternions
        """
        return Quaternion(
            self.tensor / self.modulus(),
            name=f"unit {self.name}",
            reference_frame=self.reference_frame,
        )

    def conjugate(self) -> Quaternion:
        """Return the conjugate of the quaternion"""
        return Quaternion(
            self.tensor * Tensor([-1, -1, -1, 1]),
            name=f"conjugate {self.name}",
            reference_frame=self.reference_frame,
        )

    def inverse(self) -> Quaternion:
        """Return the inverse of the quaternion"""
        return Quaternion(
            self.conjugate().tensor / self.modulus() ** 2,
            name=f"inverse {self.name}",
            reference_frame=self.reference_frame,
        )

    @staticmethod
    def product(
        q1: Quaternion,
        q2: Quaternion,
    ) -> Quaternion:
        """Return the product of two quaternions"""
        x1, y1, z1, w1 = q1.to_tuple()
        x2, y2, z2, w2 = q2.to_tuple()
        tensor = Tensor([
            x2*x1 - y2*y1 - z2*z1 - w2*w1,
            x2*y1 + y2*x1 - z2*w1 + w2*z1,
            x2*z1 + y2*w1 + z2*x1 - w2*y1,
            x2*w1 - y2*z1 + z2*y1 + w2*x1,
        ])
        return Quaternion(
            source=tensor,
            name=f"{q1.name} * {q2.name}",
            reference_frame=q1.reference_frame,
        )

    def __mul__(self, other: Quaternion) -> Quaternion:
        """Return the product of two quaternions"""
        return Quaternion.product(self, other)

    def __rmul__(self, other: Quaternion) -> Quaternion:
        """Return the product of two quaternions"""
        return Quaternion.product(other, self)

    ########################
    ###   Construction   ###
    ########################

    @classmethod
    def identity(
        cls,
        name: str = "quaternion",
        reference_frame: str | None = None,
    ) -> Quaternion:
        """Return the identity quaternion"""
        return cls(
            [0, 0, 0, 1],
            name=name,
            reference_frame=reference_frame,
        )

    @classmethod
    def from_xyzw(
        cls,
        x: float = 0,
        y: float = 0,
        z: float = 0,
        w: float = 1,
        name: str = "quaternion",
        reference_frame: str | None = None,
        unit: bool = True,
    ) -> Quaternion:
        """Create a vector from x, y, z components"""
        norm = float(np.linalg.norm([x, y, z, w]))
        if unit and norm != 1.:
            x /= norm
            y /= norm
            z /= norm
            w /= norm
        return Quaternion(
            Tensor([x, y, z, w]),
            name=name,
            reference_frame=reference_frame,
        )
