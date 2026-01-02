import os
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, PeftModel


class TextBinaryDataset(Dataset):
    """
    Generic text + binary label dataset.
    X: list[str]
    y: list[int|float] with values in {0,1}
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


@dataclass
class LoRAEstimatorConfig:
    # model/tokenizer
    model_dir: str
    model_name: str

    # training
    device: str = "cuda"
    learning_rate: float = 2e-5
    epochs: int = 1
    train_batch_size: int = 8
    eval_batch_size: int = 16
    max_length: int = 1024

    # LoRA
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj")
    bias: str = "none"

    # thresholding
    threshold: float = 0.5

    # dtype for base model weights
    torch_dtype: str = "bfloat16"  # "float16" | "float32" | "bfloat16"


def _dtype_from_string(s: str):
    s = s.lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {s}")


class LoRASequenceBinaryClassifier:
    """
    sklearn-like binary classifier:
      - fit(X, y) -> self
      - predict(X) -> np.ndarray shape (n_samples,)
      - predict_proba(X) -> np.ndarray shape (n_samples, 2)
      - decision_function(X) -> np.ndarray shape (n_samples,)
      - score(X, y) -> float accuracy
      - save(path), load(path) for adapter + metadata

    Notes:
      - Base model is loaded from (model_dir/model_name).
      - save() stores adapter weights/config; base model config may need saving separately.
    """

    def __init__(self, config: LoRAEstimatorConfig, logger=None):
        # keep init as “parameter storage” only; do not build heavy objects here
        self.config = config
        self.logger = logger

        # learned / runtime objects
        self.tokenizer_ = None
        self.model_ = None
        self.classes_ = np.array([0, 1], dtype=int)
        self.is_fitted_ = False

    def _log(self, msg: str):
        if self.logger is not None:
            self.logger.log(msg)

    def _resolve_model_path(self) -> str:
        return os.path.join(self.config.model_dir, self.config.model_name)

    def _build_tokenizer(self):
        model_path = self._resolve_model_path()
        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    def _build_model(self):
        model_path = self._resolve_model_path()
        base = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1,  # binary with BCEWithLogitsLoss
            torch_dtype=_dtype_from_string(self.config.torch_dtype),
            local_files_only=True,
        )
        base.config.pad_token_id = self.tokenizer_.pad_token_id

        peft_cfg = LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            target_modules=list(self.config.target_modules),
            lora_dropout=self.config.lora_dropout,
            bias=self.config.bias,
            task_type="SEQ_CLS",
        )
        model = get_peft_model(base, peft_cfg)
        return model.to(self.config.device)

    def fit(
        self,
        X: Sequence[str],
        y: Sequence[Union[int, float]],
        *,
        shuffle: bool = True,
    ):
        self.tokenizer_ = self._build_tokenizer()
        self.model_ = self._build_model()

        trainable_param, all_param = self.model_.get_nb_trainable_parameters()
        self._log(f"Trained {trainable_param} parameters out of {all_param} total")

        ds = TextBinaryDataset(
            X, y, tokenizer=self.tokenizer_, max_length=self.config.max_length
        )
        dl = DataLoader(ds, batch_size=self.config.train_batch_size, shuffle=shuffle)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.config.learning_rate)

        self.model_.train()
        for epoch in range(self.config.epochs):
            self._log(f"Epoch {epoch + 1} [STARTED]")
            total = 0.0

            for input_ids, attention_mask, labels in dl:
                input_ids = input_ids.to(self.config.device)
                attention_mask = attention_mask.to(self.config.device)
                labels = labels.to(self.config.device).unsqueeze(1)  # [B] -> [B,1]

                out = self.model_(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total += float(loss.item())

            self._log(f"Epoch {epoch + 1} [COMPLETED] - Avg Loss: {total / max(1, len(dl)):.4f}")

        self.is_fitted_ = True
        return self

    @torch.no_grad()
    def decision_function(self, X: Sequence[str]) -> np.ndarray:
        self._check_is_fitted()
        ds = TextBinaryDataset(X, None, tokenizer=self.tokenizer_, max_length=self.config.max_length)
        dl = DataLoader(ds, batch_size=self.config.eval_batch_size, shuffle=False)

        self.model_.eval()
        logits_all = []
        for input_ids, attention_mask in dl:
            input_ids = input_ids.to(self.config.device)
            attention_mask = attention_mask.to(self.config.device)
            out = self.model_(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits.squeeze(1)  # [B,1] -> [B]
            logits_all.append(logits.detach().cpu())

        return torch.cat(logits_all, dim=0).numpy()

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        logits = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-logits))           # sigmoid
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)

    def predict(self, X: Sequence[str]) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.config.threshold).astype(self.classes_.dtype)

    def score(self, X: Sequence[str], y: Sequence[Union[int, float]]) -> float:
        y_true = np.asarray(y).astype(int)
        y_pred = self.predict(X)
        return float((y_pred == y_true).mean())

    def save(self, output_dir: str):
        """
        Saves:
          - adapter weights/config via model.save_pretrained(output_dir)
          - tokenizer files (for reproducibility)
          - estimator metadata (init config)

        Note: PEFT save_pretrained is adapter-centric; base model config may need saving separately.
        """
        self._check_is_fitted()
        os.makedirs(output_dir, exist_ok=True)

        # adapter
        self.model_.save_pretrained(output_dir)

        # tokenizer
        tok_dir = os.path.join(output_dir, "tokenizer")
        os.makedirs(tok_dir, exist_ok=True)
        self.tokenizer_.save_pretrained(tok_dir)

        # metadata
        meta = {
            "config": asdict(self.config),
            "classes_": self.classes_.tolist(),
            "threshold": self.config.threshold,
        }
        with open(os.path.join(output_dir, "estimator.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return self

    @classmethod
    def load(
        cls,
        adapter_dir: str,
        *,
        base_model_dir: str,
        base_model_name: str,
        device: str = "cuda",
        logger=None,
    ):
        """
        Reload:
          - base model from (base_model_dir/base_model_name)
          - adapter from adapter_dir
          - tokenizer from adapter_dir/tokenizer
        """
        with open(os.path.join(adapter_dir, "estimator.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        cfg = LoRAEstimatorConfig(
            model_dir=base_model_dir,
            model_name=base_model_name,
            device=device,
            **{k: v for k, v in meta["config"].items() if k not in ("model_dir", "model_name", "device")},
        )
        est = cls(cfg, logger=logger)
        est.classes_ = np.asarray(meta.get("classes_", [0, 1]), dtype=int)

        # tokenizer
        tok_dir = os.path.join(adapter_dir, "tokenizer")
        est.tokenizer_ = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
        if est.tokenizer_.pad_token is None:
            est.tokenizer_.pad_token = est.tokenizer_.eos_token

        # base model + adapter
        base_path = os.path.join(base_model_dir, base_model_name)
        base = AutoModelForSequenceClassification.from_pretrained(
            base_path,
            num_labels=1,
            torch_dtype=_dtype_from_string(cfg.torch_dtype),
            local_files_only=True,
        )
        base.config.pad_token_id = est.tokenizer_.pad_token_id
        est.model_ = PeftModel.from_pretrained(base, adapter_dir).to(device)

        est.is_fitted_ = True
        return est

    def _check_is_fitted(self):
        if not getattr(self, "is_fitted_", False) or self.model_ is None or self.tokenizer_ is None:
            raise RuntimeError("Estimator is not fitted yet. Call fit() or load().")
