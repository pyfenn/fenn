import torch
import torch.nn as nn

from fenn.logging import Logger
from fenn.nn.trainers import ClassificationTrainer
from fenn.nn.utils import ModelPrettyPrinter


class _LargeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(32, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                    nn.ReLU(),
                )
                for _ in range(10)
            ]
        )
        self.head = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def test_model_pretty_printer_renders_full_small_model():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Sequential(nn.Linear(8, 2), nn.Sigmoid()),
    )

    rendered = ModelPrettyPrinter(model).render()

    assert "Model Summary" in rendered
    assert "-------------" in rendered
    assert "Class: Sequential" in rendered
    assert "Parameters: total=58, trainable=58, frozen=0" in rendered
    assert "Architecture" in rendered
    assert "0: Linear (params=40, in_features=4, out_features=8, bias=True)" in rendered
    assert "  1: Sigmoid" in rendered


def test_model_pretty_printer_compacts_large_model():
    model = _LargeModel()

    rendered = ModelPrettyPrinter(model, small_model_threshold=5).render()

    assert "Class: _LargeModel" in rendered
    assert "Modules: 52" in rendered
    assert "... 2 nested modules omitted" in rendered
    assert "... 2 more modules" in rendered


def test_trainer_logs_model_summary_to_logger(monkeypatch):
    logged_messages = []

    def capture(self, message, display_on_terminal=True, write_on_file=True):
        logged_messages.append(
            (message, display_on_terminal, write_on_file)
        )

    monkeypatch.setattr(Logger, "display_info", capture)

    model = nn.Sequential(nn.Linear(4, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    ClassificationTrainer(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optim=optimizer,
        num_classes=2,
    )

    model_logs = [
        entry for entry in logged_messages if entry[0].startswith("Model Summary")
    ]

    assert model_logs
    message, display_on_terminal, write_on_file = model_logs[0]
    assert "Parameters: total=10, trainable=10, frozen=0" in message
    assert display_on_terminal is False
    assert write_on_file is True
