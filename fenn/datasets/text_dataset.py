from collections.abc import Mapping, Sequence
from typing import Protocol, TypeAlias

import torch
from torch import Tensor
from torch.utils.data import Dataset

LabelValue: TypeAlias = int | float
TextDatasetItem: TypeAlias = tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]


class TextTokenizer(Protocol):
    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        padding: str,
        max_length: int,
        return_tensors: str,
    ) -> Mapping[str, Tensor]: ...


class TextDataset(Dataset[TextDatasetItem]):
    """
    Generic text + binary label dataset.
    X: list[str]
    y: list[int|float]
    """

    def __init__(
        self,
        X: Sequence[str],
        y: Sequence[LabelValue] | None,
        tokenizer: TextTokenizer,
        max_length: int = 1024,
    ) -> None:
        self.X: list[str] = list(X)
        self.y: list[float] | None = None if y is None else [float(v) for v in y]
        self.tokenizer: TextTokenizer = tokenizer
        self.max_length: int = max_length

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> TextDatasetItem:
        enc: Mapping[str, Tensor] = self.tokenizer(
            self.X[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids: Tensor = enc["input_ids"].squeeze(0)
        attention_mask: Tensor = enc["attention_mask"].squeeze(0)
        if self.y is None:
            return input_ids, attention_mask
        label: Tensor = torch.tensor(self.y[index], dtype=torch.float32)
        return input_ids, attention_mask, label
