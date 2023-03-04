from typing import Any, List, Tuple, Union
from torch import Tensor
import numpy as np
from nptyping import NDArray, Int, Float, Shape
from nptyping import DataFrame, Structure
from nptyping import assert_isinstance

# Entity
NumpyArray = NDArray[Any, Float]

# Vectors
Numpy2DVector = NDArray[Shape["2"], Float]
Numpy3DVector = NDArray[Shape["3"], Float]

# Quaternions
NumpyQuaternion = NDArray[Shape["4"], Float]

# Twists
NumpyTwist = NDArray[Shape["6"], Float]
NumpyTwistWithExtra = NDArray[Shape["7"], Float]


# Homogeneous transformation matrices
Numpy3DPose = NDArray[Shape["4, 4"], Float]
Numpy2DPose = NDArray[Shape["3, 3"], Float]

# Rotation matrices
Numpy3DRotation = NDArray[Shape["3, 3"], Float]
Numpy2DRotation = NDArray[Shape["2, 2"], Float]

# Formats that can be used to create typed entities

SourceEntity = Union[
    Tensor,
    NumpyArray,
    List[float],
    List[List[float]],
    Tuple[float, ...],
    Tuple[Tuple[float, ...], ...]
]

Source3DVector = Union[
    Tensor,
    Numpy3DVector,
    List[float],
    Tuple[float, float, float]
]

Source3DPose = Union[
    Tensor,
    Numpy3DPose,
    List[List[float]],
]

Source3DRotation = Union[
    Tensor,
    Numpy3DRotation,
    List[List[float]],
]

SourceQuaternion = Union[
    Tensor,
    NumpyQuaternion,
    List[float],
    Tuple[float, float, float, float]
]

SourceTwist = Union[
    Tensor,
    NumpyTwist,
    NumpyTwistWithExtra,
    List[float],
    Tuple[float, ...],
]

# TODO: check the linter for this
# PointsDataFrame: DataFrame[Structure["name: Str, x: Float, y: Float, z: Float"]]
