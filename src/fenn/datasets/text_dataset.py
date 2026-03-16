from typing import Optional, Sequence, Union

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Generic text + binary label dataset.
    X: list[str]
    y: list[int|float]
    """

    def __init__(
        self,
        X: Sequence[str],
        y: Optional[Sequence[Union[int, float]]],
        tokenizer,
        max_length: int = 1024,
    ):
        self.X = list(X)
        self.y = None if y is None else [float(v) for v in y]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.X[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        if self.y is None:
            return input_ids, attention_mask
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return input_ids, attention_mask, label
