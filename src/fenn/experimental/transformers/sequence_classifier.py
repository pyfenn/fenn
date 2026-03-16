import os
import json
from dataclasses import asdict
from typing import Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import peft

from fenn.core.exporter import Exporter
from fenn.logging import Logger
from fenn.datasets import TextDataset
from fenn.transformers import LoRAConfig

def _dtype_from_string(s: str):
    s = s.lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {s}")

class SequenceClassifier:
    """
    classifier:
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

    def __init__(self, config: LoRAConfig):

        self._logger = Logger()
        # keep init as “parameter storage” only; do not build heavy objects here
        self._config = config

        # learned / runtime objects
        self._tokenizer_ = None
        self._model_ = None
        self._classes_ = np.array([0, 1], dtype=int)
        self._is_trained_ = False

    def _log(self, msg: str):
        self._logger.user_info(msg)

    def _resolve_model_path(self) -> str:
        return os.path.join(self._config.model_dir, self._config.model_name)

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
            torch_dtype=_dtype_from_string(self._config.torch_dtype),
            local_files_only=True,
        )
        base.config.pad_token_id = self._tokenizer_.pad_token_id

        peft_cfg = peft.LoraConfig(
            r=self._config.r,
            lora_alpha=self._config.lora_alpha,
            target_modules=list(self._config.target_modules),
            lora_dropout=self._config.lora_dropout,
            bias=self._config.bias,
            task_type="SEQ_CLS",
        )
        model = peft.get_peft_model(base, peft_cfg)
        return model.to(self._config.device)

    def fit(
        self,
        X: Sequence[str],
        y: Sequence[Union[int, float]],
        *,
        shuffle: bool = True,
    ):
        self._tokenizer_ = self._build_tokenizer()
        self._model_ = self._build_model()

        trainable_param, all_param = self._model_.get_nb_trainable_parameters()
        self._log(f"Trained {trainable_param} parameters out of {all_param} total")

        ds = TextDataset(
            X, y, tokenizer=self._tokenizer_, max_length=self._config.max_length
        )
        dl = DataLoader(ds, batch_size=self._config.train_batch_size, shuffle=shuffle)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self._model_.parameters(), lr=self._config.learning_rate)

        self._model_.train()
        for epoch in range(self._config.epochs):
            self._log(f"Epoch {epoch + 1} [STARTED]")
            total = 0.0

            for input_ids, attention_mask, labels in dl:
                input_ids = input_ids.to(self._config.device)
                attention_mask = attention_mask.to(self._config.device)
                labels = labels.to(self._config.device).unsqueeze(1)  # [B] -> [B,1]

                out = self._model_(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total += float(loss.item())

            self._log(f"Epoch {epoch + 1} [COMPLETED] - Avg Loss: {total / max(1, len(dl)):.4f}")

        self._is_trained_ = True
        return self

    @torch.no_grad()
    def decision_function(self, X: Sequence[str]) -> np.ndarray:
        self._check_is_fitted()
        ds = TextDataset(X, None, tokenizer=self._tokenizer_, max_length=self._config.max_length)
        dl = DataLoader(ds, batch_size=self._config.eval_batch_size, shuffle=False)

        self._model_.eval()
        logits_all = []
        for input_ids, attention_mask in dl:
            input_ids = input_ids.to(self._config.device)
            attention_mask = attention_mask.to(self._config.device)
            out = self._model_(input_ids=input_ids, attention_mask=attention_mask)
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
        return (proba >= self._config.threshold).astype(self._classes_.dtype)

    def score(self, X: Sequence[str], y: Sequence[Union[int, float]]) -> float:
        y_true = np.asarray(y).astype(int)
        y_pred = self.predict(X)
        return float((y_pred == y_true).mean())

    def save(self, output_dir: str | None = None):
        """
        Saves:
          - adapter weights/config via model.save_pretrained(output_dir)
          - tokenizer files (for reproducibility)
          - estimator metadata (init config)

        Note: PEFT save_pretrained is adapter-centric; base model config may need saving separately.
        """
        self._check_is_fitted()
        if output_dir is None:
            output_dir = str(Exporter().get_export_dir("sequence_classifier"))
        os.makedirs(output_dir, exist_ok=True)

        # adapter
        self._model_.save_pretrained(output_dir)

        # tokenizer
        tok_dir = os.path.join(output_dir, "tokenizer")
        os.makedirs(tok_dir, exist_ok=True)
        self._tokenizer_.save_pretrained(tok_dir)

        # metadata
        meta = {
            "config": asdict(self._config),
            "classes_": self._classes_.tolist(),
            "threshold": self._config.threshold,
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
        device: str = "cuda"
    ):
        """
        Reload:
          - base model from (base_model_dir/base_model_name)
          - adapter from adapter_dir
          - tokenizer from adapter_dir/tokenizer
        """
        with open(os.path.join(adapter_dir, "estimator.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        cfg = LoRAConfig(
            model_dir=base_model_dir,
            model_name=base_model_name,
            device=device,
            **{k: v for k, v in meta["config"].items() if k not in ("model_dir", "model_name", "device")},
        )

        est = cls(cfg)
        est._classes_ = np.asarray(meta.get("classes_", [0, 1]), dtype=int)

        # tokenizer
        tok_dir = os.path.join(adapter_dir, "tokenizer")
        est._tokenizer_ = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
        if est._tokenizer_.pad_token is None:
            est._tokenizer_.pad_token = est._tokenizer_.eos_token

        # base model + adapter
        base_path = os.path.join(base_model_dir, base_model_name)
        base = AutoModelForSequenceClassification.from_pretrained(
            base_path,
            num_labels=1,
            torch_dtype=_dtype_from_string(cfg.torch_dtype),
            local_files_only=True,
        )
        base.config.pad_token_id = est._tokenizer_.pad_token_id
        est._model_ = peft.PeftModel.from_pretrained(base, adapter_dir).to(device)

        est._is_trained_ = True
        return est

    def _check_is_fitted(self):
        if not getattr(self, "is_fitted_", False) or self._model_ is None or self._tokenizer_ is None:
            raise RuntimeError("Estimator is not fitted yet. Call fit() or load().")
