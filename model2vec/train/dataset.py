import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(self, tokenized_texts: list[list[int]], targets: torch.Tensor, pad_id: int = 0) -> None:
        """A dataset of texts.

        :param tokenized_texts: The tokenized texts. Each text is a list of token ids.
        :param targets: The targets.
        :param pad_id: The id used to pad batches. Must match the `pad_id` of the model being trained.
        :raises ValueError: If the number of targets does not match the number of texts.
        """
        if len(targets) != len(tokenized_texts):
            raise ValueError("Number of targets does not match number of texts.")
        self.tokenized_texts = tokenized_texts
        self.targets = targets
        self.pad_id = pad_id

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.tokenized_texts)

    def __getitem__(self, index: int) -> tuple[list[int], torch.Tensor]:
        """Gets an item."""
        return self.tokenized_texts[index], self.targets[index]

    def collate_fn(self, batch: list[tuple[list[list[int]], int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate function."""
        texts, targets = zip(*batch)

        tensors: list[torch.Tensor] = [torch.LongTensor(x) for x in texts]
        padded = pad_sequence(tensors, batch_first=True, padding_value=self.pad_id)

        return padded, torch.stack(targets)

    def to_dataloader(self, shuffle: bool, batch_size: int = 32) -> DataLoader:
        """Convert the dataset to a DataLoader."""
        return DataLoader(self, collate_fn=self.collate_fn, shuffle=shuffle, batch_size=batch_size)


class PairDataset(Dataset):
    def __init__(
        self,
        tokenized_texts_a: list[list[int]],
        tokenized_texts_b: list[list[int]],
        labels: list[int] | torch.Tensor | None = None,
        pad_id: int = 0,
    ) -> None:
        """A dataset of aligned text pairs.

        :param tokenized_texts_a: The tokenized first half of each pair. Each text is a list of token ids.
        :param tokenized_texts_b: The tokenized second half of each pair. Each text is a list of token ids.
        :param labels: The label for each pair: 1 if the pair should be pushed together, 0 if it should be
            pushed towards a cosine similarity of 0. If None, every pair is labeled 1.
        :param pad_id: The id used to pad batches. Must match the `pad_id` of the model being trained.
        :raises ValueError: If the two halves don't have the same number of texts, or if `labels` doesn't
            have one entry per pair.
        """
        if len(tokenized_texts_a) != len(tokenized_texts_b):
            raise ValueError("The two halves of a pair dataset must have the same number of texts.")
        if labels is not None and len(labels) != len(tokenized_texts_a):
            raise ValueError("labels must have one entry per pair.")
        self.tokenized_texts_a = tokenized_texts_a
        self.tokenized_texts_b = tokenized_texts_b
        self.labels = torch.ones(len(tokenized_texts_a)) if labels is None else torch.as_tensor(labels).float()
        self.pad_id = pad_id

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.tokenized_texts_a)

    def __getitem__(self, index: int) -> tuple[list[int], list[int], torch.Tensor]:
        """Gets an item."""
        return self.tokenized_texts_a[index], self.tokenized_texts_b[index], self.labels[index]

    def collate_fn(self, batch: list[tuple[list[int], list[int], torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate function.

        Both halves are padded together so they end up with the same sequence length, then
        stacked into a single (2, batch_size, seq_len) tensor.
        """
        texts_a, texts_b, labels = zip(*batch)

        tensors: list[torch.Tensor] = [torch.LongTensor(x) for x in (*texts_a, *texts_b)]
        padded = pad_sequence(tensors, batch_first=True, padding_value=self.pad_id)
        padded_a, padded_b = padded[: len(texts_a)], padded[len(texts_a) :]

        return torch.stack([padded_a, padded_b]), torch.stack(labels)

    def to_dataloader(self, shuffle: bool, batch_size: int = 32) -> DataLoader:
        """Convert the dataset to a DataLoader."""
        return DataLoader(self, collate_fn=self.collate_fn, shuffle=shuffle, batch_size=batch_size)
