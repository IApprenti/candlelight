from typing import Dict, Tuple, List

# Vocabulary
# TODO: sort that alphabetically

UID: str = "unique-identifier"
STRING: str = "string"
NAME: str = "name"
ROOT: str = "root"
TRAIN: str = "train"
TEST: str = "test"
VALIDATION: str = "validation"
VAL: str = VALIDATION
GROUND_TRUTH: str = "ground-truth"
MAPPING: str = "mapping"
MASK: str = "mask"
SPLIT: str = "split"
TEXT: str = "text"
LABEL: str = "label"
LABELS: str = "labels"
ENCODINGS: str = "encodings"
TARGETS: str = "targets"
PREDICTIONS: str = "predictions"
EMBEDDINGS: str = "embeddings"
MAPPINGS: str = "mappings"
WEIGHTS: str = "weights"
CLASSES: str = "classes"
SCORES: str = "scores"
SOURCE: str = "source"
TARGET: str = "target"
INFO: str = "info"
CLASSIFICATION: str = "classification"
REGRESSION: str = "regression"
CLUSTERING: str = "clustering"
LOCALISATION: str = "localisation"
SEGMENTATION: str = "segmentation"
ANOMALY_DETECTION: str = "anomaly detection"
EMBEDDING: str = "embedding"
ENCODING: str = "encoding"
PREDICTION: str = "prediction"
WEIGHT: str = "weight"
CLASS: str = "class"
SCORE: str = "score"

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
VECTOR_2D: str = "vector2d"
VECTOR_3D: str = "vector3d"
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
    VECTOR_2D: (2,),
    VECTOR_3D: (3,),
    QUATERNION: (4,),
    ROTATION_2D: (2, 2),
    ROTATION_3D: (3, 3),
    TRANSFORM_2D: (3, 3),
    TRANSFORM_3D: (4, 4),
}

RPY_CONVENTION = "xyz"

# Extensions

NUMPY_EXTENSIONS: Tuple[str,...] = ("npy", "npz", "np")
NUMPY_COMPATIBLE_EXTENSIONS: Tuple[str,...] = ("npy", "npz", "csv", "txt", "tsv", "xls", "xlsx")
IMAGE_EXTENSIONS: Tuple[str,...] = ("png", "jpg", "jpeg", "bmp", "tif", "tiff")
VIDEO_EXTENSIONS: Tuple[str,...] = ("mp4", "avi", "mov", "mkv")
AUDIO_EXTENSIONS: Tuple[str,...] = ("mp3", "wav", "ogg")
TENSOR_EXTENSIONS: Tuple[str,...] = ("pt", "pth")
PICKLE_EXTENSIONS: Tuple[str,...] = ("pkl", "pickle")
TEXT_EXTENSIONS: Tuple[str,...] = ("txt", "csv")
JSON_EXTENSIONS: Tuple[str,...] = ("json",)
YAML_EXTENSIONS: Tuple[str,...] = ("yaml", "yml")
EXCEL_EXTENSIONS: Tuple[str,...] = ("xls", "xlsx", "xlsm", "xlsb")
DATAFRAME_EXTENSIONS: Tuple[str,...] = ("csv", "tsv", "xls", "xlsx", "xlsm", "xlsb", "odf", "ods", "odt")
SERIALIZABLE_EXTENSIONS: Tuple[str,...] = ("json", "yaml", "yml", "pkl", "pickle")
FEATHER_EXTENSIONS: Tuple[str,...] = ("feather",)
PARQUET_EXTENSIONS: Tuple[str,...] = ("parquet",)
HDF_EXTENSIONS: Tuple[str,...] = ("hdf", "hdf5", "h5")