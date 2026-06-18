from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pytest

from fenn.nn import Checkpoint, ClassificationTrainer

from sklearn.metrics import accuracy_score

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not avialable"
)

class LeNet(nn.Module):
    """
    A simple CNN for image classification.
    Default architecture is designed for 32x32 RGB images (e.g., CIFAR-10).
    """

    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # After 3 pooling layers: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Block 2: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Block 3: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
    

#unit test : just the model

def test_lenet_output_shape():
    model = LeNet(in_channels=3, num_classes=10)
    x = torch.randn(4,3,32,32)
    out = model(x)
    assert out.shape == (4,10)

def test_lenet_gradient():
    model = LeNet(in_channels=3, num_classes=10)
    x = torch.randn(4,3,32,32)
    out = model(x)
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters())

@cuda_only
def test_gpu_forward():
    # 1. Initialize and move to GPU
    model = LeNet(in_channels=3, num_classes=10).cuda()
    
    # CRITICAL: Put the model in eval mode to turn off Dropout randomness!
    model.eval() 
    
    # 2. Create input tensor directly on the GPU device
    x = torch.randn(4, 3, 32, 32, device="cuda")
    
    # 3. Forward pass on GPU
    out = model(x)
    
    # 4. Assertions
    assert out.device.type == "cuda"
    
    # Move the model to CPU for the baseline verification
    # They will match now because dropout is frozen
    assert torch.allclose(out.cpu(), model.cpu()(x.cpu()), atol=1e-4)
    
#smoke test: training loop, no CIFAR-10

def make_fake_loader(n=8, batch=4):
    x = torch.randn(n, 3, 32, 32)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch)

def test_trainer_smoke():
    model = LeNet(in_channels=3, num_classes=10)
    trainer = ClassificationTrainer(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optim=optim.Adam(model.parameters(), lr=1e-3),
        num_classes=10,
        device="cpu"
    )
    trainer.fit(train_loader=make_fake_loader(), epochs=2, val_loader=make_fake_loader())

def test_predict_returns_labels():
    model = LeNet(in_channels=3, num_classes=10)
    trainer = ClassificationTrainer(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optim=optim.Adam(model.parameters(), lr=1e-3),
        num_classes=10,
        device="cpu",
    )
    trainer.fit(train_loader=make_fake_loader(), epochs=1, val_loader=make_fake_loader())
    preds = trainer.predict(make_fake_loader())
    assert len(preds) == 8
    assert all(0 <= p < 10 for p in preds)