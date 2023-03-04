import torch
from torch import Tensor
from ..types_alias import SourceEntity
from .entity import Entity

# TODO: strong typing to rich types


def dot(
    entity1: Entity,
    entity2: Entity
) -> Tensor | float:
    """Dot product between two torch based entities"""
    result: Tensor = entity1.tensor.dot(entity2.tensor)
    if len(result) == 1:
        return result.item()
    return result


def cross(
    entity1: Entity,
    entity2: Entity
) -> Tensor:
    """Cross product between two torch based entities"""
    return entity1.tensor.cross(entity2.tensor)


def add(
    entity1: Entity,
    entity2: Entity
) -> Tensor:
    """Add two torch based entities"""
    return entity1.tensor + entity2.tensor


def sub(
    entity1: Entity,
    entity2: Entity
) -> Tensor:
    """Subtract two torch based entities"""
    return entity1.tensor - entity2.tensor
