from scipy.spatial.transform import Rotation as R
from .rotation import Rotation
from .vector import Vector
from .quaternion import Quaternion
from .euler_angles import EulerAngles


# To matrix

def vector_to_matrix(vector: Vector) -> Rotation:
    """Convert a rotation vector to a rotation matrix"""
    scipy = R.from_rotvec(vector.to_tuple())
    return Rotation(
        scipy.as_matrix(),
        name=f"Rotation matrix from rotation vector {vector.name}",
        reference_frame=vector.reference_frame,
    )


def quaternion_to_matrix(quaternion: Quaternion) -> Rotation:
    """Convert a quaternion to a rotation matrix"""
    scipy = R.from_quat(quaternion.to_tuple())
    return Rotation(
        scipy.as_matrix(),
        name=f"Rotation matrix from quaternion {quaternion.name}",
        reference_frame=quaternion.reference_frame,
    )


def euler_angles_to_matrix(euler_angles: EulerAngles) -> Rotation:
    """Convert a rotation vector to a rotation matrix"""
    scipy = R.from_euler(
        euler_angles.convention,
        euler_angles.to_tuple(),
        degrees=False,
    )
    return Rotation(
        scipy.as_matrix(),
        name=f"Rotation matrix from euler angles {euler_angles.name}",
        reference_frame=euler_angles.reference_frame,
    )

# From matrix


def matrix_to_vector(matrix: Rotation) -> Vector:
    """Convert a rotation matrix to a rotation vector"""
    scipy = R.from_matrix(matrix.tensor)
    return Vector(
        scipy.as_rotvec(),
        name=f"Rotation vector from rotation matrix {matrix.name}",
        reference_frame=matrix.reference_frame,
    )


def matrix_to_quaternion(matrix: Rotation) -> Quaternion:
    """Convert a rotation matrix to a quaternion"""
    scipy = R.from_matrix(matrix.tensor)
    return Quaternion(
        scipy.as_quat(),
        name=f"Quaternion from rotation matrix {matrix.name}",
        reference_frame=matrix.reference_frame,
    )


def matrix_to_euler_angles(matrix: Rotation, convention: str = EulerAngles.RPY_CONVENTION) -> EulerAngles:
    """Convert a rotation matrix to euler angles"""
    scipy = R.from_matrix(matrix.tensor)
    return EulerAngles(
        scipy.as_euler(
            convention,
            degrees=False,
        ),
        name=f"Euler angles from rotation matrix {matrix.name}",
        reference_frame=matrix.reference_frame,
        convention=convention
    )

# To quaternion


def vector_to_quaternion(vector: Vector) -> Quaternion:
    """Convert a rotation vector to a quaternion"""
    scipy = R.from_rotvec(vector.to_tuple())
    return Quaternion(
        scipy.as_quat(),
        name=f"Quaternion from rotation vector {vector.name}",
        reference_frame=vector.reference_frame,
    )


def euler_angles_to_quaternion(euler_angles: EulerAngles) -> Quaternion:
    """Convert a rotation vector to a quaternion"""
    scipy = R.from_euler(
        euler_angles.convention,
        euler_angles.to_tuple(),
        degrees=False,
    )
    return Quaternion(
        scipy.as_quat(),
        name=f"Quaternion from euler angles {euler_angles.name}",
        reference_frame=euler_angles.reference_frame,
    )

# From quaternion


def quaternion_to_vector(quaternion: Quaternion) -> Vector:
    """Convert a quaternion to a rotation vector"""
    scipy = R.from_quat(quaternion.to_tuple())
    return Vector(
        scipy.as_rotvec(),
        name=f"Rotation vector from quaternion {quaternion.name}",
        reference_frame=quaternion.reference_frame,
    )


def quaternion_to_euler_angles(
    quaternion: Quaternion,
    convention: str = EulerAngles.RPY_CONVENTION
) -> EulerAngles:
    """Convert a quaternion to euler angles"""
    scipy = R.from_quat(quaternion.to_tuple())
    return EulerAngles(
        scipy.as_euler(
            convention,
            degrees=False,
        ),
        name=f"Euler angles from quaternion {quaternion.name}",
        reference_frame=quaternion.reference_frame,
        convention=convention
    )

# To euler angles


def vector_to_euler_angles(
    vector: Vector,
    convention: str = EulerAngles.RPY_CONVENTION
) -> EulerAngles:
    """Convert a rotation vector to euler angles"""
    scipy = R.from_rotvec(vector.to_tuple())
    return EulerAngles(
        scipy.as_euler(
            convention,
            degrees=False,
        ),
        name=f"Euler angles from rotation vector {vector.name}",
        reference_frame=vector.reference_frame,
        convention=convention
    )

# From euler angles


def euler_angles_to_vector(
    euler_angles: EulerAngles
) -> Vector:
    """Convert euler angles to a rotation vector"""
    scipy = R.from_euler(
        euler_angles.convention,
        euler_angles.to_tuple(),
        degrees=False,
    )
    return Vector(
        scipy.as_rotvec(),
        name=f"Rotation vector from euler angles {euler_angles.name}",
        reference_frame=euler_angles.reference_frame,
    )
