from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from datasets import Column
from datasets import Dataset as HFDataset
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

LABEL_COLUMN = "label"
TEXT_COLUMN = "text"
TEXT_A_COLUMN = "text_a"
TEXT_B_COLUMN = "text_b"


def _loader_generators() -> tuple[torch.Generator, torch.Generator]:
    """Create a generator for a sampler and one for its DataLoader, both seeded from the global torch RNG."""
    sampler_seed, loader_seed = torch.randint(2**62, (2,)).tolist()
    return torch.Generator().manual_seed(sampler_seed), torch.Generator().manual_seed(loader_seed)


def remove_unk(token_ids: list[int], unk_token_id: int | None) -> list[int]:
    """Drop unknown tokens, mirroring `StaticModel.tokenize`."""
    if unk_token_id is None:
        return token_ids
    return [token_id for token_id in token_ids if token_id != unk_token_id]


class BatchTokenizer:
    def __init__(self, tokenizer: Tokenizer, unk_token_id: int | None, max_length: int | None) -> None:
        """Tokenizes batches of texts into lists of token ids. Can be pickled, e.g. to be sent to DataLoader workers.

        :param tokenizer: The tokenizer. It should not pad.
        :param unk_token_id: The id of the unknown token, which is dropped. If None, no tokens are dropped.
        :param max_length: The maximum length of the input in tokens. If this is None, no truncation is done.
        """
        self.tokenizer = tokenizer
        self.unk_token_id = unk_token_id
        self.max_length = max_length

    def __call__(self, texts: list[str]) -> list[list[int]]:
        """Tokenize a batch of texts."""
        if self.max_length is not None:
            truncate_length = self.max_length * 10
            texts = [text[:truncate_length] for text in texts]
        encoded = self.tokenizer.encode_batch_fast(texts, add_special_tokens=False)
        return [remove_unk(encoding.ids, self.unk_token_id)[: self.max_length] for encoding in encoded]


def float_targets(labels: list[Any]) -> torch.Tensor:
    """Turn a batch of numeric labels, such as vectors, into a float tensor."""
    return torch.as_tensor(labels, dtype=torch.float32)


class ClassTargets:
    def __init__(self, classes: Sequence[Hashable], multilabel: bool) -> None:
        """Turns a batch of class labels into a tensor of targets. Can be pickled, e.g. to be sent to DataLoader workers.

        :param classes: The classes, in order.
        :param multilabel: Whether each label is a list of classes, turned into a multi-hot vector. Otherwise, each
            label is a single class, turned into its index.
        """
        self.index = {label: index for index, label in enumerate(classes)}
        self.multilabel = multilabel

    def __call__(self, labels: list[Any]) -> torch.Tensor:
        """Turn a batch of labels into targets.

        :param labels: The labels. A tensor or an array is converted to a list first.
        :return: The class indices, or the multi-hot vectors if the task is multilabel.
        :raises ValueError: If a label is not one of the classes.
        """
        if isinstance(labels, (torch.Tensor, np.ndarray)):
            labels = labels.tolist()
        try:
            if not self.multilabel:
                return torch.tensor([self.index[label] for label in labels], dtype=torch.long)
            targets = torch.zeros(len(labels), len(self.index), dtype=torch.float)
            for row, sample_labels in enumerate(labels):
                if isinstance(sample_labels, (torch.Tensor, np.ndarray)):
                    sample_labels = sample_labels.tolist()
                targets[row, [self.index[label] for label in sample_labels]] = 1.0
            return targets
        except KeyError as error:
            raise ValueError(f"Label {error.args[0]!r} is not one of the classes {list(self.index)}.") from None


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


def read_column(column: Column) -> pa.ChunkedArray:
    """Read a whole column of a Hugging Face dataset as an Arrow array, without converting it to Python objects.

    :param column: The column, which can be nested, such as `dataset["metadata"]["label"]`. Its dataset must not
        have a transform.
    :return: The values of the column.
    """
    dataset, name, fields = _column_path(column)
    return _struct_fields(dataset.select_columns([name]).with_format("arrow")[:].column(name), fields)


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
    return array.null_count == 0 or read_column(column).null_count == 0


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


class _Batches(Dataset):
    def __init__(self, rows: ColumnRows, indices: np.ndarray | None, pad_id: int) -> None:
        """A dataset that fetches rows and turns them into items per batch.

        :param rows: The rows to draw items from.
        :param indices: The indices of the rows that belong to this dataset. If None, all rows belong to it.
        :param pad_id: The id used to pad batches. Must match the `pad_id` of the model being trained.
        """
        self.rows = rows
        self.indices = np.arange(len(rows)) if indices is None else indices
        self.pad_id = pad_id

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        """Gets an item."""
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: list[int]) -> list[Any]:
        """Fetch and convert a batch of items at once."""
        return self._to_items(self.rows[self.indices[indices].tolist()])

    def _to_items(self, rows: Mapping[str, Any]) -> list[Any]:
        """Turn a batch of rows into items."""
        raise NotImplementedError

    def collate_fn(self, batch: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate a batch of items into model inputs and targets."""
        raise NotImplementedError

    def _drop_last(self, batch_size: int) -> bool:
        """Whether to drop the last batch if it is smaller than `batch_size`."""
        return False

    def to_dataloader(self, shuffle: bool, batch_size: int = 32, num_workers: int = 0) -> DataLoader:
        """Convert the dataset to a DataLoader, which loads batches in `num_workers` worker processes.

        The order of the batches does not depend on `num_workers`.
        """
        sampler_generator, loader_generator = _loader_generators()
        sampler = RandomSampler(self, generator=sampler_generator) if shuffle else SequentialSampler(self)
        return DataLoader(
            self,
            collate_fn=self.collate_fn,
            batch_sampler=BatchSampler(sampler, batch_size=batch_size, drop_last=self._drop_last(batch_size)),
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            generator=loader_generator,
        )


class TextDataset(_Batches):
    def __init__(
        self,
        rows: ColumnRows,
        tokenize: Callable[[list[str]], list[list[int]]],
        to_targets: Callable[[Any], torch.Tensor],
        indices: np.ndarray | None = None,
        pad_id: int = 0,
    ) -> None:
        """A dataset of labeled texts, which are tokenized per batch.

        :param rows: The labeled texts, in a `text` and a `label` column.
        :param tokenize: Turns a batch of texts into lists of token ids.
        :param to_targets: Turns a batch of labels into a tensor of targets.
        :param indices: The indices of the rows that belong to this dataset. If None, all rows belong to it.
        :param pad_id: The id used to pad batches. Must match the `pad_id` of the model being trained.
        """
        super().__init__(rows, indices, pad_id)
        self.tokenize = tokenize
        self.to_targets = to_targets

    def _to_items(self, rows: Mapping[str, Any]) -> list[tuple[list[int], torch.Tensor]]:
        """Tokenize the texts and turn the labels into targets."""
        return list(zip(self.tokenize(rows[TEXT_COLUMN]), self.to_targets(rows[LABEL_COLUMN])))

    def collate_fn(self, batch: list[tuple[list[int], torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate function."""
        texts, targets = zip(*batch)

        tensors: list[torch.Tensor] = [torch.LongTensor(x) for x in texts]
        padded = pad_sequence(tensors, batch_first=True, padding_value=self.pad_id)

        return padded, torch.stack(targets)


class PairDataset(_Batches):
    def __init__(
        self,
        rows: ColumnRows,
        tokenize: Callable[[list[str]], list[list[int]]],
        indices: np.ndarray | None = None,
        pad_id: int = 0,
    ) -> None:
        """A dataset of aligned text pairs, which are tokenized per batch.

        :param rows: The pairs, in a `text_a` and a `text_b` column.
        :param tokenize: Turns a batch of texts into lists of token ids.
        :param indices: The indices of the rows that belong to this dataset. If None, all rows belong to it.
        :param pad_id: The id used to pad batches. Must match the `pad_id` of the model being trained.
        """
        super().__init__(rows, indices, pad_id)
        self.tokenize = tokenize

    def _to_items(self, rows: Mapping[str, Any]) -> list[tuple[list[int], list[int]]]:
        """Tokenize both halves of each pair."""
        return list(zip(self.tokenize(rows[TEXT_A_COLUMN]), self.tokenize(rows[TEXT_B_COLUMN])))

    def collate_fn(self, batch: list[tuple[list[int], list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate function.

        Both halves are padded together so they end up with the same sequence length, then
        stacked into a single (2, batch_size, seq_len) tensor. The targets are the index of each
        pair's second text within the batch.
        """
        texts_a, texts_b = zip(*batch)

        tensors: list[torch.Tensor] = [torch.LongTensor(x) for x in (*texts_a, *texts_b)]
        padded = pad_sequence(tensors, batch_first=True, padding_value=self.pad_id)
        padded_a, padded_b = padded[: len(texts_a)], padded[len(texts_a) :]

        return torch.stack([padded_a, padded_b]), torch.arange(len(texts_a))

    def _drop_last(self, batch_size: int) -> bool:
        """Drop a final batch with a single pair, unless it is the only pair."""
        return len(self) > 1 and len(self) % batch_size == 1
