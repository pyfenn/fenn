try:
    import torch
except ImportError as e:
    raise RuntimeError(
        "Torch is required by smle."
        "Install it yourself (GPU/CPU) or use 'pip install smle[torch]'."
    ) from e
from pathlib import Path

class Trainer:

    def __init__(self, model, loss_fn, optimizer, device="cpu", metrics=None):
        self._device = device

        self._model = model.to(device)
        self._model.train()
        self._loss_fn = loss_fn
        self._optimizer = optimizer

        self._metrics = metrics or {}

    def fit(self, train_loader, epochs, export_dir: Path):

        for epoch in range(epochs):
            self._train_epoch(train_loader, epoch)

        save_file = export_dir / "model.pth"

        self._model.cpu()
        torch.save(self._model.state_dict(), save_file)
        self._model.to(self._device)

        return self._model

    def _train_epoch(self, loader, epoch):

        print(f"Epoch {epoch} [STARTED].")

        total_loss = 0.0
        n_batches = 0

        for batch in loader:

            inputs, targets = self._move_batch(batch)
            outputs = self._model(inputs)
            loss = self._loss_fn(outputs, targets)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        mean_loss = total_loss / n_batches
        print(f"Epoch {epoch} [ENDED]. Mean Loss: {mean_loss:.4f}")

    def _move_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            return [self._move_batch(b) for b in batch]
        if isinstance(batch, dict):
            return {k: self._move_batch(v) for k, v in batch.items()}
        if torch.is_tensor(batch):
            return batch.to(self._device)
        return batch
