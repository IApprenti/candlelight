from __future__ import annotations
from typing import Any, List
from torch import Tensor
import numpy as np
from ..types_alias import SourceEntity
from copy import copy


class Entity:
    """Define an entity in 3D
    Uses a 3D torch tensor internally to store data
    Is supposed to be subclassed to define specific entities
    """

    def __init__(
        self,
        source: SourceEntity | Entity,
        name: str = "entity",
        reference_frame: str | None = None,
        extra: Any | None = None,
    ) -> None:
        """Create a new entity"""
        if isinstance(source, Entity):
            source = source.tensor
        self.tensor: Tensor = source if isinstance(
            source, Tensor) else Tensor(source)
        self.name: str = name
        self.reference_frame: str | None = reference_frame
        self.extra: Any | None = extra

    def __str__(self) -> str:
        """Print the vector"""
        ref: str = f" in {self.reference_frame}" if self.reference_frame else ""
        return f"{self.name}{ref}: {self.tensor}"

    def __copy__(self) -> Entity:
        """Return a copy of the entity"""
        copied: Entity = copy(self)
        copied.tensor = self.tensor.clone()
        return copied

    #############################
    ###   Format conversion   ###
    #############################

    def numpy(self) -> np.ndarray:
        """Return the pose as a numpy array"""
        return self.tensor.numpy()

    def to_list(self) -> list:
        """Return the entity as a list"""
        return self.tensor.tolist()

    def to_tuple(self) -> tuple:
        """Return the entity as a tuple"""
        return tuple(self.to_list())
