

from typing import List
from torch import Tensor

class Item:

  def __init__(
    self,
    root_directory: str | None = None,
    source: str | None = None,
    split: str | None = None,
    encoding: Tensor | None = None,
    scores: Tensor | None = None,
    scores_targets: Tensor | None = None,
    scores_names: List[str] | None = None,
    classes: Tensor | None = None,
    classes_targets: Tensor | None = None,
    classes_names: List[str] | None = None,
    embedding: Tensor | None = None,
    heatmap: Tensor | None = None,
    reconstruction: Tensor | None = None,
    info: Dict[str, Any] | str | None = None,
    **kwargs : Dict[str, Any],
  ):
    self.root_directory: str | None = root_directory
    self.source: str | None = sources
    self.split: str | None = splits
    self.encoding: None = encodings
    self.scores: Tensor | None = scores
    self.scores_targets: Tensor | None = scores_targets
    self.scores_names: List[str] | None = scores_names
    self.classes: Tensor | None = classes
    self.classes_targets: Tensor | None = classes_targets
    self.classes_names: List[str] | None = scores_names
    self.embedding: Tensor | None = embeddings
    self.heatmap: Tensor | None = maps
    self.reconstruction: Tensor | None = reconstructions
    self.info: Dict[str, Any] | str | None = info
    
    # Set any additional attributes.
    for key, value in kwargs.items():
      setattr(self, key, value)
