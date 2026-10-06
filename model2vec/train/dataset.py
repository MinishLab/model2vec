from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from datasets import Column
from datasets import Dataset as HFDataset
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)

LABEL_COLUMN = "label"
TEXT_COLUMN = "text"
TEXT_A_COLUMN = "text_a"
TEXT_B_COLUMN = "text_b"


def _column_path(column: Column) -> tuple[HFDataset, str, list[str]]:
    """Find the dataset a column belongs to, its top-level column, and the struct fields leading to the column."""
    names = []
    source: Any = column
    while isinstance(source, Column):
        names.append(source.column_name)
        source = source.source
    *fields, name = names
    return source, name, fields[::-1]


def _struct_fields(array: pa.ChunkedArray, fields: list[str]) -> pa.ChunkedArray:
    """Select nested struct fields from an Arrow array."""
    for field in fields:
        array = pc.struct_field(array, field)
    return array


def has_transform(column: Column) -> bool:
    """Check whether a column belongs to a Hugging Face dataset with a transform, set with `with_transform`."""
    dataset, _, _ = _column_path(column)
    return dataset.format["type"] == "custom"


def iter_column(column: Column, batch_size: int = 10_000) -> Iterator[pa.ChunkedArray]:
    """Read a column of a Hugging Face dataset in batches of Arrow arrays, without converting it to Python objects.

    :param column: The column, which can be nested, such as `dataset["metadata"]["label"]`. Its dataset must not
        have a transform.
    :param batch_size: The number of rows in each batch.
    :return: The values of the column, one batch at a time.
    """
    dataset, name, fields = _column_path(column)
    batches = dataset.select_columns([name]).with_format("arrow").iter(batch_size=batch_size)
    return (_struct_fields(batch.column(name), fields) for batch in batches)


def column_type(column: Column) -> pa.DataType:
    """Get the Arrow type of a column of a Hugging Face dataset, without reading the column.

    :param column: The column, which can be nested, such as `dataset["metadata"]["label"]`.
    :return: The type of the values of the column.
    """
    dataset, name, fields = _column_path(column)
    return _struct_fields(dataset.data.column(name), fields).type


def has_only_strings(column: Column) -> bool:
    """Check that a column of a Hugging Face dataset only holds strings, reading the strings only if it has nulls.

    :param column: The column, which can be nested, such as `dataset["metadata"]["text"]`. Its dataset must not
        have a transform.
    :return: Whether every row of the column is a string.
    """
    dataset, name, fields = _column_path(column)
    array = _struct_fields(dataset.data.column(name), fields)
    if not (pa.types.is_string(array.type) or pa.types.is_large_string(array.type)):
        return False
    return array.null_count == 0 or not any(batch.null_count for batch in iter_column(column))


def read_label_column(labels: Column, name: str) -> tuple[bool, Counter]:
    """Determine whether a column of labels is multi-label, and count the number of times each class occurs.

    :param labels: A column of a Hugging Face dataset. If it holds lists, multi-label classification is assumed.
    :param name: The name of the labels, used in error messages.
    :return: Whether the labels are multi-label, and the number of times each class occurs.
    :raises ValueError: If the labels are not strings, integers, or lists of those, or if a label is missing.
    """
    label_type = column_type(labels)
    multilabel = (
        pa.types.is_list(label_type) or pa.types.is_large_list(label_type) or pa.types.is_fixed_size_list(label_type)
    )
    value_type = label_type.value_type if multilabel else label_type
    if not (pa.types.is_string(value_type) or pa.types.is_large_string(value_type) or pa.types.is_integer(value_type)):
        raise ValueError(f"Labels in {name} must be strings, integers, or lists of those, got {label_type}.")
    counts: Counter = Counter()
    for array in iter_column(labels):
        values = pc.list_flatten(array) if multilabel else array
        if array.null_count or values.null_count:
            raise ValueError(f"Labels in {name} must not be missing.")
        value_counts = pc.value_counts(values)
        counts.update(dict(zip(value_counts.field("values").to_pylist(), value_counts.field("counts").to_pylist())))
    return multilabel, counts


def get_vector_dims_from_column(vectors: Column, name: str) -> set[int]:
    """Get the dimensions of the vectors in a column of a Hugging Face dataset, reading the column in batches.

    :param vectors: A column of a Hugging Face dataset that holds lists of numbers.
    :param name: The name of the vectors, used in error messages.
    :return: The dimensions found, stopping as soon as more than one is found.
    :raises ValueError: If a vector is missing, or if the column doesn't hold lists of numbers.
    """
    array_type = column_type(vectors)
    is_list = pa.types.is_list(array_type) or pa.types.is_large_list(array_type)
    if not (is_list or pa.types.is_fixed_size_list(array_type)):
        raise ValueError(f"{name} must hold lists of numbers, got {array_type}.")
    value_type = array_type.value_type
    if not (pa.types.is_floating(value_type) or pa.types.is_integer(value_type)):
        raise ValueError(f"{name} must hold lists of numbers, got {array_type}.")
    dims: set[int] = set()
    for array in iter_column(vectors):
        if array.null_count or pc.list_flatten(array).null_count:
            raise ValueError(f"Vectors in {name} must not be missing.")
        bounds = pc.min_max(pc.list_value_length(array))
        dims |= {bounds["min"].as_py(), bounds["max"].as_py()}
        if len(dims) > 1:
            break
    return dims


def _list_strata(labels: Sequence[Any]) -> list[np.ndarray]:
    """Group the indices of a list of labels by label, in order of first occurrence."""
    indices_by_label: dict[Any, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        indices_by_label[label].append(index)
    return [np.asarray(indices) for indices in indices_by_label.values()]


def _array_strata(labels: np.ndarray) -> list[np.ndarray]:
    """Group the indices of an array of labels by label, in order of first occurrence."""
    _, first, codes = np.unique(labels, return_index=True, return_inverse=True)
    codes = np.argsort(np.argsort(first))[codes]
    return np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1])


def _column_strata(labels: Column) -> list[np.ndarray]:
    """Group the indices of a Hugging Face column of single labels by label, in order of first occurrence.

    :param labels: The labels, which must be strings or integers, none of them missing.
    :return: The indices of each label.
    """
    classes: dict[Any, int] = {}
    batch_codes = []
    for array in iter_column(labels):
        encoded = pc.dictionary_encode(array.combine_chunks())
        mapping = np.array([classes.setdefault(label, len(classes)) for label in encoded.dictionary.to_pylist()])
        batch_codes.append(mapping[encoded.indices.to_numpy(zero_copy_only=False)])
    codes = np.concatenate(batch_codes)
    return np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1])


def stratify_indices(labels: Sequence[Any]) -> list[np.ndarray] | None:
    """Group the indices of single labels by label, or return None if there are no labels or a label occurs once.

    :param labels: Single labels that have been validated: strings or integers, none of them missing.
    :return: The indices of each label, in order of first occurrence, or None if the labels can't be stratified.
    """
    if not len(labels):
        return None
    if isinstance(labels, Column):
        strata = _column_strata(labels)
    elif isinstance(labels, torch.Tensor):
        strata = _array_strata(labels.cpu().numpy())
    elif isinstance(labels, np.ndarray):
        strata = _array_strata(labels)
    else:
        strata = _list_strata(labels)
    if min(len(indices) for indices in strata) < 2:
        logger.info("Some classes have fewer than 2 samples. Stratification is disabled.")
        return None
    return strata


class ColumnRows:
    def __init__(self, **columns: Sequence[Any] | torch.Tensor) -> None:
        """Rows made up of aligned columns, which are only read for the rows that are fetched.

        :param **columns: The values of each column. A tensor, an array, or a column of a Hugging Face dataset is
            indexed with a list of indices at once. Any other sequence is indexed item by item.
        :raises ValueError: If the columns don't all have the same length.
        """
        lengths = {name: len(column) for name, column in columns.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"All columns must have the same length, got {lengths}.")
        self.columns: dict[str, Any] = columns

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(next(iter(self.columns.values())))

    def __getitem__(self, indices: list[int]) -> dict[str, Any]:
        """Fetch the rows at the given indices, as a mapping from column names to the values of those rows."""
        return {
            name: column[indices]
            if isinstance(column, (np.ndarray, torch.Tensor, Column))
            else [column[index] for index in indices]
            for name, column in self.columns.items()
        }


@dataclass(frozen=True)
class TokenBatch:
    """A batch of token id sequences, stored as one flat tensor of ids with the offset and length of each sequence."""

    ids: torch.Tensor
    offsets: torch.Tensor
    lengths: torch.Tensor

    @classmethod
    def from_lengths(cls, ids: torch.Tensor, lengths: torch.Tensor) -> TokenBatch:
        """Create a batch from a flat tensor of ids and the length of each sequence."""
        return cls(ids, lengths.cumsum(0) - lengths, lengths)

    @classmethod
    def from_token_ids(cls, token_ids: Sequence[Sequence[int]]) -> TokenBatch:
        """Flatten lists of token ids into a batch."""
        lengths = np.fromiter(map(len, token_ids), dtype=np.int64, count=len(token_ids))
        ids = np.fromiter(chain.from_iterable(token_ids), dtype=np.int64, count=int(lengths.sum()))
        return cls.from_lengths(torch.from_numpy(ids), torch.from_numpy(lengths))

    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.lengths)

    def to(self, device: torch.device | str) -> TokenBatch:
        """Move the batch to a device."""
        return TokenBatch(self.ids.to(device), self.offsets.to(device), self.lengths.to(device))


@dataclass(frozen=True)
class PairBatch:
    """A batch of pairs: the first texts followed by the second texts, and an id per text that is shared by identical texts."""

    tokens: TokenBatch
    text_ids: torch.Tensor

    def to(self, device: torch.device | str) -> PairBatch:
        """Move the batch to a device."""
        return PairBatch(self.tokens.to(device), self.text_ids.to(device))


class _Batches(Dataset, ABC):
    def __init__(self, rows: ColumnRows, indices: np.ndarray | None) -> None:
        """A dataset that fetches rows and turns them into items per batch.

        :param rows: The rows to draw items from.
        :param indices: The indices of the rows that belong to this dataset. If None, all rows belong to it.
        """
        self.rows = rows
        self.indices = np.arange(len(rows)) if indices is None else indices

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        """Gets an item."""
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: list[int]) -> list[Any]:
        """Fetch and convert a batch of items at once."""
        return self._to_items(self.rows[self.indices[indices].tolist()])

    @abstractmethod
    def _to_items(self, rows: Mapping[str, Any]) -> list[Any]:
        """Turn a batch of rows into items."""

    @abstractmethod
    def collate_fn(self, batch: list[Any]) -> tuple[TokenBatch | PairBatch, torch.Tensor]:
        """Collate a batch of items into model inputs and targets."""

    def _drop_last(self, batch_size: int) -> bool:
        """Whether to drop the last batch if it is smaller than `batch_size`."""
        return False

    def to_dataloader(self, shuffle: bool, batch_size: int = 32) -> DataLoader:
        """Convert the dataset to a DataLoader."""
        sampler = RandomSampler(self) if shuffle else SequentialSampler(self)
        return DataLoader(
            self,
            collate_fn=self.collate_fn,
            batch_sampler=BatchSampler(sampler, batch_size=batch_size, drop_last=self._drop_last(batch_size)),
        )


class TextDataset(_Batches):
    def __init__(
        self,
        rows: ColumnRows,
        tokenize: Callable[[list[str]], list[list[int]]],
        to_targets: Callable[[Any], torch.Tensor],
        indices: np.ndarray | None = None,
    ) -> None:
        """A dataset of labeled texts, which are tokenized per batch.

        :param rows: The labeled texts, in a `text` and a `label` column.
        :param tokenize: Turns a batch of texts into lists of token ids.
        :param to_targets: Turns a batch of labels into a tensor of targets.
        :param indices: The indices of the rows that belong to this dataset. If None, all rows belong to it.
        """
        super().__init__(rows, indices)
        self.tokenize = tokenize
        self.to_targets = to_targets

    def _to_items(self, rows: Mapping[str, Any]) -> list[tuple[list[int], torch.Tensor]]:
        """Tokenize the texts and turn the labels into targets."""
        return list(zip(self.tokenize(rows[TEXT_COLUMN]), self.to_targets(rows[LABEL_COLUMN])))

    def collate_fn(self, batch: list[tuple[list[int], torch.Tensor]]) -> tuple[TokenBatch, torch.Tensor]:
        """Collate function."""
        texts, targets = zip(*batch)
        return TokenBatch.from_token_ids(texts), torch.stack(targets)


class PairDataset(_Batches):
    def __init__(
        self,
        rows: ColumnRows,
        tokenize: Callable[[list[str]], list[list[int]]],
        indices: np.ndarray | None = None,
    ) -> None:
        """A dataset of aligned text pairs, which are tokenized per batch.

        :param rows: The pairs, in a `text_a` and a `text_b` column.
        :param tokenize: Turns a batch of texts into lists of token ids.
        :param indices: The indices of the rows that belong to this dataset. If None, all rows belong to it.
        """
        super().__init__(rows, indices)
        self.tokenize = tokenize

    def _to_items(self, rows: Mapping[str, Any]) -> list[tuple[list[int], list[int]]]:
        """Tokenize both halves of each pair."""
        return list(zip(self.tokenize(rows[TEXT_A_COLUMN]), self.tokenize(rows[TEXT_B_COLUMN])))

    def collate_fn(self, batch: list[tuple[list[int], list[int]]]) -> tuple[PairBatch, torch.Tensor]:
        """Collate function.

        The first texts and the second texts are put into a single batch, first texts first. The targets
        are the index of each pair's second text within the batch.
        """
        texts_a, texts_b = zip(*batch)
        texts = (*texts_a, *texts_b)
        ids_by_text: dict[tuple[int, ...], int] = {}
        text_ids = [ids_by_text.setdefault(tuple(text), len(ids_by_text)) for text in texts]

        pair_batch = PairBatch(TokenBatch.from_token_ids(texts), torch.tensor(text_ids, dtype=torch.int64))
        return pair_batch, torch.arange(len(texts_a))

    def _drop_last(self, batch_size: int) -> bool:
        """Drop a final batch with a single pair, unless it is the only pair."""
        return len(self) > 1 and len(self) % batch_size == 1
