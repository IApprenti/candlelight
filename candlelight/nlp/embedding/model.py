"""Create sentence embeddings from text."""
from typing import Any, List
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from torch import Tensor
from .embedding import Embedding


class Model:
    """Sentence embedding model."""

    def __init__(
        self,
        model_name: str,
        output_dim: int,
        max_seq_length: int | None = None,
        size_in_mb: int | None = None,
        languages: List[str] | None = None,
        device: str = "cpu",
    ):
        """Initialize model."""
        self.model_name: str = model_name
        self.output_dim: int = output_dim
        self.max_seq_length: int | None = max_seq_length
        self.size_in_mb: int | None = size_in_mb
        self.languages: List[str] | None = languages
        self.device: str = device
        #: Container to store a third-party model.
        self._model: Any | None = None

    def embed(
        self,
        sentences: List[str]
    ) -> Embedding:
        """Create sentence embeddings."""
        return Embedding(
            sentences=sentences,
            tensor=self.encode(sentences),
            model_name=self.model_name,
        )

    def encode(
        self,
        sentences: List[str]
    ) -> Tensor:
        """Create sentence embeddings."""
        raise NotImplementedError

    def can_speak(self, language: str) -> bool | None:
        """Return True if model can speak the language."""
        if self.languages is None:
            return None
        return language in self.languages
