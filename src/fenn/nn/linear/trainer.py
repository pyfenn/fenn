import torch
#from pathlib import Path

from fenn.logging import Logger

class Trainer:

    def __init__(self,
                 model,
                 loss_fn,
                 optim,
                 epochs,
                 device="cpu"):

        self._logger = Logger()

        self._device = device

        self._model = model.to(device)
        self._model.train()
        self._loss_fn = loss_fn
        self._optimizer = optim
        self._epochs = epochs
        self._metrics = {}

    def fit(self, train_loader):

        for epoch in range(self._epochs):
            self._logger.system_info(f"Epoch {epoch} started.")

            total_loss = 0.0
            n_batches = 0

            for data, labels in train_loader:
                data = data.to(self._device)
                labels = labels.to(self._device)

                outputs = self._model(data)
                loss = self._loss_fn(outputs, labels)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            mean_loss = total_loss / n_batches
            print(f"Epoch {epoch}. Mean Loss: {mean_loss:.4f}")
        #save_file = export_dir / "model.pth"
        #self._model.cpu()
        #torch.save(self._model.state_dict(), save_file)
        #self._model.to(self._device)

        return self._model

    def _move_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            return [self._move_batch(b) for b in batch]
        if isinstance(batch, dict):
            return {k: self._move_batch(v) for k, v in batch.items()}
        if torch.is_tensor(batch):
            return batch.to(self._device)
        return batch
