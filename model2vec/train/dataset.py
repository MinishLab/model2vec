import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(self, tokenized_texts: list[list[int]], targets: torch.Tensor) -> None:
        """
        A dataset of texts.

        :param tokenized_texts: The tokenized texts. Each text is a list of token ids.
        :param targets: The targets.
        :raises ValueError: If the number of targets does not match the number of texts.
        """
        if len(targets) != len(tokenized_texts):
            raise ValueError("Number of targets does not match number of texts.")
        self.tokenized_texts = tokenized_texts
        self.targets = targets

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.tokenized_texts)

    def __getitem__(self, index: int) -> tuple[list[int], torch.Tensor]:
        """Gets an item."""
        return self.tokenized_texts[index], self.targets[index]

    @staticmethod
    def collate_fn(batch: list[tuple[list[list[int]], int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate function."""
        texts, targets = zip(*batch)

        tensors: list[torch.Tensor] = [torch.LongTensor(x) for x in texts]
        padded = pad_sequence(tensors, batch_first=True, padding_value=0)

        return padded, torch.stack(targets)

    def to_dataloader(self, shuffle: bool, batch_size: int = 32) -> DataLoader:
        """Convert the dataset to a DataLoader."""
        return DataLoader(self, collate_fn=self.collate_fn, shuffle=shuffle, batch_size=batch_size)
