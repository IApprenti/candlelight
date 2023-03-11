from typing import List
from .model import Model
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch.nn.modules.module import Module


class SBert(Model):
    """Sentence embedding model."""

    def get_model(self) -> SentenceTransformer:
        """Get the sentence transformer model."""
        self._model: SentenceTransformer = SentenceTransformer(
            self.model_name,
            device=self.device,
        )
        return self._model

    def get_max_seq_length(self):
        """Get the maximum sequence length."""
        if self._model is None:
            self.get_model()
        length: Tensor | Module = self._model.max_seq_length
        self.max_seq_length = int(length)
        return self.max_seq_length

    def set_max_seq_length(self, max_seq_length: int):
        """Set the maximum sequence length."""
        self.max_seq_length = max_seq_length
        self._model.max_seq_length = max_seq_length

    def encode(self, sentences: List[str]) -> Tensor:
        """Create sentence embeddings."""
        if self._model is None:
            self.get_model()
        embeddings = self._model.encode(
            sentences,
            show_progress_bar=False,
            convert_to_tensor=True,
        )
        if not isinstance(embeddings, Tensor):
            raise ValueError("Expected embeddings to be a torch.Tensor")
        return embeddings
