from __future__ import annotations
from typing import List, Tuple
from torch import Tensor
import torch
import pickle
import os


class Embedding:
    """Embedding of a list of sentences.
    """

    def __init__(
        self,
        sentences: List[str],
        tensor: Tensor,
        model_name: str,
        language: str | None = None,
    ):
        """Initialize embedding."""
        self.sentences: List[str] = sentences
        self.tensor: Tensor = tensor
        self.model_name: str = model_name
        self.language: str | None = language

    def __str__(self) -> str:
        """Return string representation."""
        return f"Embedding(model_name={self.model_name}, num_sentences={len(self)})"

    def __repr__(self) -> str:
        """Return string representation."""
        return str(self)

    #################
    ###   Shape   ###
    #################

    @property
    def shape(self) -> Tuple[int, int]:
        """Return shape of embedding tensor."""
        return self.tensor.shape

    @property
    def input_dim(self) -> int:
        """Return embedding dimension."""
        return self.tensor.shape[1]

    def __dim__(self) -> int:
        """Return embedding dimension."""
        return self.tensor.shape[1]

    def __len__(self) -> int:
        """Return number of of embedding."""
        return self.tensor.shape[0]

    ####################
    ###   Indexing   ###
    ####################

    def __call__(self, index: int | str) -> Tuple[str, Tensor]:
        """Return the original sentence and embedding tensor."""
        if isinstance(index, str):
            index = self.get_index(index)
        return self.sentences[index], self.tensor[index]

    def __getitem__(self, index: int) -> Tensor:
        """Return embedding value at index."""
        return self.tensor[index]

    def get_sentence(self, index: int) -> str:
        """Return sentence at index."""
        return self.sentences[index]

    def get_index(self, sentence: str) -> int:
        """Return the index of a sentence."""
        return self.sentences.index(sentence)

    @property
    def first(self) -> Tuple[str, Tensor]:
        """Return the first sentence and embedding tensor."""
        return self(0)

    @property
    def last(self) -> Tuple[str, Tensor]:
        """Return the last sentence and embedding tensor."""
        return self(-1)

    def __iter__(self):
        """Iterate over embeddings."""
        for i in range(len(self)):
            yield self[i]

    ##################################
    ###   Mix several embeddings   ###
    ##################################

    def __add__(self, other: Embedding) -> Embedding:
        """Add two embeddings."""
        return Embedding(
            sentences=self.sentences + other.sentences,
            tensor=torch.concat((self.tensor, other.tensor), dim=0),
            model_name=self.model_name,
        )

    ##################
    ###   Pickle   ###
    ##################

    def save(self, dir_path: str, file_name: str = "embeddings"):
        """Store sentences & embeddings on disc"""
        file: str = f"{file_name}.pkl"
        path: str = os.path.join(dir_path, file)
        with open(path, "wb") as f_out:
            pickle.dump(
                {
                    'sentences': self.sentences,
                    'embeddings': self.tensor,
                    'model_name': self.model_name,
                    'language': self.language,
                },
                f_out,
                protocol=pickle.HIGHEST_PROTOCOL
            )

    @classmethod
    def load(
        cls,
        dir_path: str,
        file_name: str = "embeddings"
    ):
        """Create an Embedding object from stored data"""
        file: str = f"{file_name}.pkl"
        path: str = os.path.join(dir_path, file)
        with open(path, "rb") as f_in:
            stored_data = pickle.load(f_in)
            stored_sentences = stored_data['sentences']
            stored_embeddings = stored_data['embeddings']
            stored_model_name = stored_data['model_name']
            stored_language = stored_data['language']
        return cls(
            sentences=stored_sentences,
            tensor=stored_embeddings,
            model_name=stored_model_name,
            language=stored_language,
        )
