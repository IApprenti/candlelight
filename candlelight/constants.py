from typing import Dict, Tuple

# Coordinate system

X_INDEX: int = 0
Y_INDEX: int = 1
Z_INDEX: int = 2
W_INDEX: int = 3

X: str = "x"
Y: str = "y"
Z: str = "z"
W: str = "w"

index_names: Dict[int, str] = {
    X_INDEX: X,
    Y_INDEX: Y,
    Z_INDEX: Z,
    W_INDEX: W,
}

AXIS: str = "axis"

# Entities

SCALAR: str = "scalar"
VECTOR: str = "vector"
MATRIX: str = "matrix"
QUATERNION: str = "quaternion"
S3D: str = "3d"
S2D: str = "2d"
VECTOR2D: str = "vector2d"
VECTOR3D: str = "vector3d"
TRANSFORM_3D: str = "homogeneous transformation matrix 3D"
ROTATION_3D: str = "rotation matrix 3D"
TRANSFORM_2D: str = "homogeneous transformation matrix 2D"
ROTATION_2D: str = "rotation matrix 2D"

structure_dimensions: Dict[str, int] = {
    SCALAR: 0,
    VECTOR: 1,
    MATRIX: 2,
    QUATERNION: 1,
}

structure_shape: Dict[str, Tuple[int, ...]] = {
    SCALAR: (1,),
    VECTOR2D: (2,),
    VECTOR3D: (3,),
    QUATERNION: (4,),
    ROTATION_2D: (2, 2),
    ROTATION_3D: (3, 3),
    TRANSFORM_2D: (3, 3),
    TRANSFORM_3D: (4, 4),
}

RPY_CONVENTION = "xyz"
