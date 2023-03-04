import torch
from scipy.spatial.transform import Rotation as R
from .rotation import Rotation
from .vector import Vector
from .quaternion import Quaternion
from .euler_angles import EulerAngles
from .pose import Pose
from .twist import Twist
from .convert_rotation import matrix_to_euler_angles, euler_angles_to_matrix


def pose_to_twist(
    pose: Pose,
    convention: str = Twist.RPY_CONVENTION,
) -> Twist:
    """Convert a pose to a twist"""
    translation: Vector = pose.translation
    rotation: Rotation = pose.rotation
    euler_angles: EulerAngles = matrix_to_euler_angles(rotation, convention)
    return Twist(
        source=torch.concat((translation.tensor, euler_angles.tensor)),
        name=f"Twist from pose {pose.name}",
        reference_frame=pose.reference_frame,
        extra=pose.extra,
    )


def twist_to_pose(
    twist: Twist,
    convention: str = Twist.RPY_CONVENTION,
) -> Pose:
    """Convert a twist to a pose"""
    translation: Vector = Vector(twist.tensor[:3])
    euler_angles: EulerAngles = EulerAngles(
        twist.tensor[3:],
        convention=convention,
    )
    rotation: Rotation = euler_angles_to_matrix(euler_angles)
    return Pose.from_translation_and_rotation(
        translation=translation,
        rotation=rotation,
        name=f"Pose from twist {twist.name}",
        reference_frame=twist.reference_frame,
        extra=twist.extra,
    )
