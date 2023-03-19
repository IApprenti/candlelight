from __future__ import annotations
from typing import Any, Dict, List
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pickle
from .item import Item

class Data(Dataset):
    """Store several items of data.
    Have the ability to encode items and provide datasets/dataloaders for the splits.
    Can be inherited to create custom items such as images, text embeddings, etc.
    """

    def __init__(
            self,
            root_directory: str | None = None,
            splits: List[str] | None = None,
            sources: List[str] | None = None,
            encodings: Tensor | None = None,
            targets: Tensor | Dict[str, Tensor] | None = None,
            predictions: Tensor | Dict[str, Tensor] | None = None,
            tensors: Dict[str, Tensor] | None = None,  # Find a better name for this, we store score, loss, embeddings, etc.
            allow_encoding_all: bool = False,
            store_encodings: bool = False,
            cursor: int = 0,
            info: dict[str, str] | str | None = None,
            **kwargs: Dict[str, Any]
    ) -> None:        
        self.root_directory: str | None = root_directory
        self.sources: List[str] | None = sources
        self.splits: List[str] | None = splits
        self._encodings: Tensor | None = encodings
        self.targets: Tensor | Dict[str, Tensor] | None = targets
        self.predictions: Tensor | Dict[str, Tensor] | None = predictions
        self.tensors: Dict[str, Tensor] | None = tensors
        self.cursor: int = cursor
        self.allow_encoding_all: bool = allow_encoding_all
        self.store_encodings: bool = store_encodings
        self.info: dict[str, str] | str | None = info

        for key, value in kwargs.items():
          setattr(self, key, value)

    ################################################
    ###   Interoperability with the Item class   ###
    ################################################

    def item(self, index: int | None = None) -> Item:
        """Return an item for a given index.
        """
        if index is None:
            index = self.cursor
        return Item(
            root_directory=self.root_directory,
            split=self.splits[index] if self.splits is not None else None,
            source=self.sources[index] if self.sources is not None else None,
            encoding=self.encodings[index] if self.encodings is not None else None,
            index=index,
            info=self.info
        )
    
    def items(self) -> List[Item]:
        """Return a list of items.
        """
        return [self.item(index) for index in range(len(self))]

    def item_from_source(self, source: str) -> Item | None:
        """Return an item for a given source.
        """
        if self.sources is None:
            return None
        return self.item(self.sources.index(source))

    def append_item(self, item: Item) -> None:
        """Append an item to the data"""
        if self.sources is not None and item.source is not None:
            self.sources.append(item.source)
        if self.splits is not None and item.split is not None:
            self.splits.append(item.split)
        if self._encodings is not None and self.we_have_all_encodings and item.encoding is not None:
            self._encodings = torch.cat([self._encodings, item.encoding])

    def append_items(self, items: List[Item]) -> None:
        """Append a list of items to the data"""
        for item in items:
            self.append_item(item)

    @property
    def we_have_all_encodings(self) -> bool:
        """Check if we have all the encodings.
        """
        if self._encodings is None:
            return False
        if self.sources is None:
            return True
        return len(self.sources) == self._encodings.shape[0]

    ##################################
    ###  Pytorch data management   ###
    ##################################

    def __len__(self) -> int:
        if self.sources is not None:
            return len(self.sources)
        if self._encodings is not None:
            return self._encodings.shape[0]
        return 0
    
    def __getitem__(self, index: int) -> Tensor | None:
        if self._encodings is not None and index < self._encodings.shape[0]:
            return self._encodings[index]
        elif self.sources is not None and index < len(self.sources):
            return self.encode(self.sources[index])
        return None
    
    def dataset(self, split: str | None = None) -> Dataset:
        """Return a dataset for a given split.
        """
        if split is None or self.splits is None:
            return self
        # TODO

    def dataloader(
            self,
            split: str | None = None,
            batch_size: int = 1,
            shuffle: bool = False,
            num_workers: int = 0
        ) -> DataLoader:
        """Return a dataloader for a given split.
        """
        if split is None or self.splits is None:
            return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        # TODO

    ####################
    ###   Encoding   ###
    ####################

    ###   Get / set all encodings   ###

    @property
    def encodings(self) -> Tensor | None:
        """Get the encodings for all items.
        """
        # If we have all encodings, return them
        if self._encodings is not None and (
            self.sources is None 
            or (
                self.sources is not None 
                and self._encodings.shape[0] == len(self.sources)
            )
        ):
            return self._encodings
        # If we are not allowed to generate all the encodings, just return what we have
        if not self.allow_encoding_all:
            # If we have nothing, return nothing
            if self._encodings is None:
                return None
            # If we have some encodings, return them
            return self._encodings
        # If we are allowed to generate all the encodings, do it
        if self.sources is not None and self._encodings is None:
            # Encode the all the item
            encodings = self.batch_encode(self.sources)
            # Store the encodings if we are allowed to
            if self.store_encodings:
                self._encodings = encodings
            # Return the encodings
            return encodings
        elif self.sources is not None and self._encodings is not None:
            # Encode the missing items
            # The missing items are a batch of items starting from the last index
            start_index = self._encodings.shape[0]
            end_index = len(self.sources)
            missing_sources = self.sources[start_index:end_index]
            missing_encodings = self.batch_encode(missing_sources)
            # Concatenate the missing encodings to the existing ones
            if missing_encodings is not None:
                encodings = torch.cat([self._encodings, missing_encodings], dim=0)
                if self.store_encodings:
                    self._encodings = encodings
                return encodings
            else:
                return self._encodings
        else:
            return self._encodings

    @encodings.setter
    def encodings(self, encodings: Tensor) -> None:
        self._encodings = encodings

    @encodings.deleter
    def encodings(self) -> None:
        self._encodings = None

    ###    Get the encoding for a single item   ###

    def get_encoding(self, index: int | None = None) -> Tensor | None:
        """Get the encoding for a single item."""
        if index is None:
            index = self.cursor
        if self._encodings is not None:
            existing_encoding = self._encodings[index] if index < len(self._encodings) else None
            if existing_encoding is not None:
                return existing_encoding
        if self.sources is not None:
            source = self.sources[index]
            if source is not None:
                encoding = self.encode(source)
                # We can only store the encoding if it is the next missing encoding
                if self.encodings is not None and encoding is not None and self.encodings.shape[0] == index and self.store_encodings:
                    self.encodings = torch.cat([self.encodings, encoding])
                return encoding
        else:
            return None
        
    @property
    def encoding(self) -> Tensor | None:
        """Get the encoding for a single item at the current cursor."""
        return self.get_encoding()

    ###   Encode / Decode functions to be reimplemented  ###
    
    def encode(self, source: str) -> Tensor | None:
        """Encode one item.
        """
        return None
    
    def batch_encode(self, sources: List[str]) -> Tensor | None:
        """Encode the items in batches.
        Reimplement if a more efficient method is available.
        """
        # Encode the first item
        encoded = self.encode(sources[0])
        if encoded is not None:
            encodings = torch.unsqueeze(encoded, dim=0)
        else:
            self.logger.warning("Failed to encode the first item, the whoole batch will be discarded.")
            return None 
        # Encode the remaining items
        for index, source in enumerate(sources):
            encoded = self.encode(source)
            if encoded is not None:
                encodings = torch.cat((encodings, encoded))
            else:
                self.logger.warning(f"Failed to encode item {index}, the whoole batch will be discarded.")
                return None
    
    ################
    ###   Show   ###
    ################

    def show(self, index: int | None = None) -> None:
        """Show the item at the given index.
        """
        if index is None:
            index = self.cursor
        if self.sources is not None:
            source = self.sources[index]
            if source is not None:
                self.show_source(source)

    def show_source(self, source: str) -> None:
        """Show the source.
        """
        print(source)

    def show_encoding(self, index: int | None = None) -> None:
        """Show the encoding for the item at the given index.
        """
        if index is None:
            index = self.cursor
        if self.encodings is not None:
            encoding = self.encodings[index]
            if encoding is not None:
                self.show_encoding(encoding)

    #######################
    ###   Save / Load   ###
    #######################

    # Use Pickle to save the dataset by default but can be reimplemented

    def save(self, path: str) -> None:
        """Save the dataset to a file.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> Data:
        """Load a dataset from a file.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
        