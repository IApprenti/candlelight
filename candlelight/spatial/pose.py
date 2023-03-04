from __future__ import annotations
from typing import Any, List
import torch
from torch import Tensor
import numpy as np
from scipy.spatial.transform import Rotation as R
from candlelight.spatial.quaternion import Quaternion
from ..types_alias import Source3DPose, Source3DVector
from .vector import Vector
from .entity import Entity
from ..constants import index_names
from .. import constants as c

from .rotation import Rotation


class Pose(Entity):
    """Define a frame position and orientation in 3D
    Uses a 4x4 homogeneous transformation internally matrix to represent the pose
    """

    def __init__(
        self,
        source: Source3DPose | Pose,
        name: str = "pose",
        reference_frame: str | None = None,
        extra: Any | None = None,
    ) -> None:
        """Create a new pose"""
        if isinstance(source, Pose):
            if extra is None:
                extra = source.extra
            source = source.tensor.clone()
        super().__init__(source, name, reference_frame, extra)

    ######################
    ###   Components   ###
    ######################

    ###  Translation   ###

    TRANSLATION_COLUMN_INDEX: int = 3

    @property
    def x(self) -> float:
        """Return the x component of the translation"""
        return self.tensor[c.X_INDEX, self.TRANSLATION_COLUMN_INDEX].item()

    @x.setter
    def x(self, x: float) -> None:
        """Set the x component of the translation"""
        self.tensor[c.X_INDEX, self.TRANSLATION_COLUMN_INDEX] = x

    @property
    def y(self) -> float:
        """Return the y component of the translation"""
        return self.tensor[c.X_INDEX, self.TRANSLATION_COLUMN_INDEX].item()

    @y.setter
    def y(self, y: float) -> None:
        """Set the y component of the translation"""
        self.tensor[c.Y_INDEX, self.TRANSLATION_COLUMN_INDEX] = y

    @property
    def z(self) -> float:
        """Return the z component of the translation"""
        return self.tensor[c.Z_INDEX, self.TRANSLATION_COLUMN_INDEX].item()

    @z.setter
    def z(self, z: float) -> None:
        """Set the z component of the translation"""
        self.tensor[c.Z_INDEX, self.TRANSLATION_COLUMN_INDEX] = z

    @property
    def translation(self) -> Vector:
        """Return the translation component of the pose"""
        return Vector(
            source=self.tensor[:3, self.TRANSLATION_COLUMN_INDEX],
            name=f"{self.name} 's translation",
            reference_frame=self.reference_frame
        )

    @translation.setter
    def translation(self, translation: Source3DVector | Vector) -> None:
        """Set the translation component of the pose"""
        if isinstance(translation, Vector):
            translation = translation.tensor
        elif isinstance(translation, Tensor) and translation.shape == torch.Size([3]):
            self.tensor[:3, self.TRANSLATION_COLUMN_INDEX] = translation
        elif isinstance(translation, (np.ndarray, list, tuple)) and len(translation) == 3:
            self.tensor[:3, self.TRANSLATION_COLUMN_INDEX] = Tensor(
                translation)
        else:
            raise TypeError(
                f"Cannot set translation from {type(translation)}: {translation}")

    @property
    def homogeneous_translation(self) -> Tensor:
        """Return the homogeneous translation component of the pose"""
        return self.tensor[:, self.TRANSLATION_COLUMN_INDEX]

    @homogeneous_translation.setter
    def homogeneous_translation(self, translation: Tensor) -> None:
        """Set the homogeneous translation component of the pose"""
        self.tensor[:, self.TRANSLATION_COLUMN_INDEX] = translation

    ###  Rotation   ###

    def rotation_tensor(self) -> Tensor:
        """Return the rotation matrix of the pose"""
        return self.tensor[:3, :3]

    @property
    def rotation(self) -> Rotation:
        """Return the rotation component of the pose"""
        return Rotation(
            source=self.rotation_tensor(),
            name=f"{self.name}'s rotation",
            reference_frame=self.reference_frame
        )

    @rotation.setter
    def rotation(self, rotation: Rotation) -> None:
        """Set the rotation component of the pose"""
        if isinstance(rotation, Rotation):
            rotation = Rotation(rotation)
        self.tensor[:3, :3] = rotation.tensor

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

    ###   roll pitch yaw rotation   ###

    def euler_angles(self, degrees: bool = False, convention=c.RPY_CONVENTION) -> tuple[float, float, float]:
        """Return the roll, pitch, and yaw of the rotation"""
        scipy_rotation = R.from_matrix(self.rotation_tensor().numpy())
        euler_angles = scipy_rotation.as_euler(
            convention, degrees=degrees)
        return tuple(euler_angles)

    def roll(self, degrees: bool = False) -> float:
        """Return the roll of the rotation"""
        return self.euler_angles(degrees=degrees)[0]

    def pitch(self, degrees: bool = False) -> float:
        """Return the pitch of the rotation"""
        return self.euler_angles(degrees=degrees)[1]

    def yaw(self, degrees: bool = False) -> float:
        """Return the yaw of the rotation"""
        return self.euler_angles(degrees=degrees)[2]

    ###   x y z translation   ###

    def translation_tuple(self, millimeters: bool = False) -> tuple[float, float, float]:
        """Return the translation as a tuple"""
        if millimeters:
            return tuple(self.translation.tensor.numpy() * 1000)
        return tuple(self.translation.tensor.numpy())

    ###  Quaternion  ###

    @property
    def quaternion(self) -> Quaternion:
        """Return a quaternion"""
        quat = R.from_matrix(self.rotation_tensor().numpy()).as_quat()
        return Quaternion(
            quat,
            name=f"{self.name}'s quaternion",
            reference_frame=self.reference_frame,
            extra=self.extra,
        )

    ######################
    ###   Operations   ###
    ######################

    def __dot__(self, other: Pose) -> Pose:
        """Multiply two poses together"""
        return Pose(self.tensor @ other.tensor)

    def __mul__(self, other: Pose) -> Pose:
        """Multiply two poses together"""
        return self.__dot__(other)

    def __rmul__(self, other: Pose) -> Pose:
        """Multiply two poses together"""
        return other.__dot__(self)

    ########################
    ###   Constructors   ###
    ########################

    @classmethod
    def identity(
        cls,
        name: str = "pose",
        reference_frame: str | None = None,
        extra: Any | None = None,
    ) -> Pose:
        """Return the identity pose"""
        return cls(
            torch.eye(4),
            name=name,
            reference_frame=reference_frame,
            extra=extra,
        )

    @classmethod
    def from_translation_and_rotation(
        cls,
        translation: Vector,
        rotation: Rotation,
        name: str | None = None,
        reference_frame: str | None = None,
        extra: Any | None = None,
    ) -> Pose:
        """Return a pose from a translation and rotation"""
        use_name = name if name is not None else f"pose from {translation.name} and {rotation.name}"
        pose = cls.identity(use_name, reference_frame, extra)
        pose.translation = translation
        pose.rotation = rotation
        if pose.extra is None and translation.extra is not None:
            pose.extra = translation.extra
        if pose.extra is None and rotation.extra is not None:
            pose.extra = rotation.extra
        return pose

    ###########################################################
    ###   Conversion to common third party data structures  ###
    ###########################################################

    def to_scipy_rotation(self) -> R:
        """Return a scipy Rotation instance"""
        return R.from_matrix(self.rotation_tensor().numpy())

    def to_translation_and_quaternion_lists(
        self,
        w_at_the_start: bool = False,
    ) -> tuple[List[float], List[float]]:
        """Return a translation and quaternion"""
        quaternion = self.quaternion.to_list()
        if w_at_the_start:
            quaternion = [
                quaternion[3],
                quaternion[0],
                quaternion[1],
                quaternion[2]
            ]
        return self.translation.to_list(), quaternion

    def to_twist_list(
        self,
        millimeters: bool = False,
        degrees: bool = False,
        convention=c.RPY_CONVENTION,
        include_extra: bool = False,
    ) -> list:
        """
        Shortcut to return a twist as a simple list instead of a Twist object
        Twists are a common format for representing poses
        This provides a list of [x, y, z, roll, pitch, yaw]
        The euler angles default convention is xyz extrinsic
        None SI units are supported as these are often used
        """
        x, y, z = self.translation_tuple(millimeters=millimeters)
        roll, pitch, yaw = self.euler_angles(
            degrees=degrees, convention=convention)
        if include_extra:
            return [x, y, z, roll, pitch, yaw, self.extra]
        return [x, y, z, roll, pitch, yaw]

    @classmethod
    def from_twist_list(
        cls,
        twist_list: list | tuple | np.ndarray | Tensor,
        millimeters: bool = False,
        degrees: bool = False,
        convention=c.RPY_CONVENTION,
        reference_frame: str | None = None,
        include_extra: bool = True,
    ):
        """Create a pose from a raw twist
        The twist is given as a simple list, tuple, array or tensor"""
        x, y, z = twist_list[0:3]
        roll, pitch, yaw = twist_list[3:6]
        extra: Any | None = None
        if len(twist_list) == 6:
            extra = twist_list[6]
        if len(twist_list) > 6:
            extra = twist_list[6:]
        if millimeters:
            x, y, z = float(x) / 1000, float(y) / 1000, float(z) / 1000
        translation = Vector([float(x), float(y), float(z)])
        scipy_rotation = R.from_euler(
            seq=convention,
            angles=[roll, pitch, yaw],
            degrees=degrees,
        )
        rotation = Rotation(scipy_rotation.as_matrix())
        return cls.from_translation_and_rotation(
            translation=translation,
            rotation=rotation,
            name=f"pose from twist {twist_list}",
            reference_frame=reference_frame,
            extra=extra if include_extra else None,
        )
