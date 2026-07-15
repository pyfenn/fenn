import torch
import torch.nn as nn

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
    assert "Modules: 53" in rendered
    assert "... 2 more modules" in rendered
