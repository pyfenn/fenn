# Binary MLP Template

## Overview

The Binary MLP (Multi-Layer Perceptron) template is designed for **binary classification tasks**. It demonstrates how to:
- Build a simple neural network for tabular/structured data
- Handle binary classification with custom datasets
- Train using the FENN framework with validation and early stopping
- Evaluate model performance on binary classification metrics

This template uses the Iris dataset as an example, classifying whether a flower is a Setosa or not.

## Architecture

### BinaryMLP Model

A simple 3-layer feedforward neural network optimized for binary classification:

```
Input (4 features)
    ↓
Linear(4 → 18) → ReLU
    ↓
Linear(18 → 16) → ReLU
    ↓
Linear(16 → 1)
    ↓
Output (1 logit for binary classification)
```

**Architecture Details:**
- **Layer 1**: 4 input features → 18 hidden units with ReLU activation
- **Layer 2**: 18 → 16 hidden units with ReLU activation
- **Output Layer**: 16 → 1 unit (single logit for binary classification)

**Loss Function**: BCEWithLogitsLoss (Binary Cross-Entropy with Logits)

### Dataset

The template includes a custom `BinaryDataset` class that:
- Takes numpy arrays (features X and labels y)
- Returns torch tensors for training
- Automatically reshapes labels to [B, 1] format for BCEWithLogitsLoss

## Configuration

The template uses YAML configuration for hyperparameters:

### Example fenn.yaml

```yaml
project: mlp

logger:
  dir: logger

general:
  device: cuda

train:
  seed: 42
  batch: 16
  epochs: 30
  lr: 1e-3

test:
  size: 0.2
  batch: 16
```

### Configuration Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `general.device` | str | Device to use (cpu, cuda) | cuda |
| `train.seed` | int | Random seed for reproducibility | 42 |
| `train.batch` | int | Batch size for training | 16 |
| `train.epochs` | int | Maximum number of training epochs | 30 |
| `train.lr` | float | Learning rate for optimizer | 0.001 |
| `test.size` | float | Proportion of data for testing (0-1) | 0.2 |
| `test.batch` | int | Batch size for testing | 16 |


## Key Features

### Binary Classification
- Uses BCEWithLogitsLoss which combines sigmoid and binary cross-entropy
- Output is a single logit that gets converted to probability during prediction
- Predictions use threshold of 0.5 for binary labels

### Validation & Early Stopping
- Validates every epoch on the test set during training
- Automatically stops after 5 epochs without improvement (early stopping)
- Saves the best model to `./checkpoints/checkpoint_best.pt`
- During validation, computes: accuracy, precision, recall, F1-score

**Validation Process:**
```python
# Train with validation on test set
model = trainer.fit(
  train_loader=train_loader,
  val_loader=test_loader,  # Validation set
)
```

The trainer automatically monitors validation loss to:
1. Determine the best model (saves when val_loss improves)
2. Trigger early stopping (stops if val_loss doesn't improve for 5 epochs)
3. Report validation metrics (accuracy, precision, recall, F1)

### Data Preprocessing
- **Feature Scaling**: StandardScaler normalizes input features
- **Train-Test Split**: 80% training, 20% testing with stratification
- **Custom Dataset**: BinaryDataset handles tensor conversion

### Reproducibility
- Fixed random seed ensures consistent results
- Stratified split maintains class distribution

## Customization

### Using Different Data

To use your own dataset instead of Iris:

```python
# Load your own CSV
import pandas as pd

df = pd.read_csv("your_data.csv")
X = df.drop("target", axis=1).values
y = df["target"].values.astype(np.int64)
```

### Adjusting Model Architecture

To modify the MLP layers:

```python
class CustomBinaryMLP(nn.Module):
    def __init__(self, input_size=4, hidden_sizes=[32, 16]):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### Adjusting Hyperparameters

Modify `fenn.yaml`:

```yaml
train:
  seed: 42
  batch: 32        # Increase batch size
  epochs: 50       # Train longer
  lr: 5e-4         # Lower learning rate
```

## Performance Monitoring

The trainer automatically logs:
- Training loss per epoch
- Validation accuracy, precision, recall, F1-score
- Early stopping events
- Best model checkpoint location

Example output:
```
Epoch 0. Mean Loss: 0.6931
Epoch 5. Validation Loss: 0.6543
Epoch 5. Validation Accuracy: 0.8333
Epoch 5. Validation Precision: 0.8500
Epoch 5. Validation Recall: 0.8000
Epoch 5. Validation F1 Score: 0.8235
```

## Expected Results

On the Iris binary classification task (Setosa vs Others):
- Training accuracy: ~98%
- Test accuracy: ~95-98%

Results depend on:
- Random seed
- Train-test split
- Hyperparameters

## File Structure

```
mlp-binary/
├── main.py                    # Training script
├── modules/
│   ├── __init__.py           # Module imports
│   ├── model.py              # BinaryMLP architecture
│   └── dataset.py            # BinaryDataset class
├── fenn.yaml                 # Configuration file
├── requirements.txt          # Python dependencies
└── mlp_binary.md            # This documentation
```

## Dependencies

Required packages:
- PyTorch
- scikit-learn (for preprocessing and metrics)
- numpy
- pandas
- ucimlrepo (for fetching UCI datasets)
- FENN framework

Install with:
```bash
pip install torch scikit-learn numpy pandas ucimlrepo
```

## Running the Template

```bash
# Using FENN CLI
python main.py

# With custom config
FENN_CONFIG=custom.yaml python main.py

# Check available devices
python -c "import torch; print(torch.cuda.is_available())"
```

## Binary Classification Metrics

The trainer automatically computes during validation:

- **Accuracy**: Proportion of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

For binary classification, these metrics are particularly important:
- High precision: few false positives (important for medical diagnosis)
- High recall: few false negatives (important for disease screening)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `batch` size in fenn.yaml |
| Slow training | Enable GPU: set `device: cuda` |
| Poor accuracy | Try lower learning rate (5e-4) or more epochs |
| Overfitting | Add regularization or reduce model complexity |
| NaN loss | Check data normalization and learning rate |

## References

- [PyTorch Binary Classification](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- FENN Framework Documentation
