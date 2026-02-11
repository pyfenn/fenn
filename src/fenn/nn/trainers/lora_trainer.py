from typing import Optional, Union, List
import torch
from pathlib import Path
from fenn.nn.trainers.trainer import Trainer

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None # type: ignore

class LoRATrainer(Trainer):
    """
    LoRATrainer extends the base Trainer to support Parameter-Efficient Fine-Tuning (PEFT)
    using LoRA (Low-Rank Adaptation).
    """

    def __init__(self,
                 model,
                 loss_fn,
                 optim,
                 epochs,
                 lora_config: Optional['LoraConfig'] = None,
                 device="cpu",
                 checkpoint_dir: Optional[Union[Path, str]] = None, 
                 checkpoint_epochs: Optional[Union[int, List[int]]] = None,
                 checkpoint_name: str = "lora_checkpoint",
                 save_best: bool = False
                 ):
        """
        Initialize the LoRATrainer.

        Args:
            model: The base model to be fine-tuned.
            loss_fn: The loss function.
            optim: The optimizer.
            epochs: Number of training epochs.
            lora_config: configuration for LoRA. If None, it assumes the model is already a PEFT model or 
                         ignores PEFT wrapping (behaves like standard Trainer).
            device: Device to train on.
            checkpoint_dir: Directory to save checkpoints.
            checkpoint_epochs: Epochs at which to save checkpoints.
            checkpoint_name: Base name for checkpoint files.
            save_best: Whether to save the best model based on loss.
        """
        
        if lora_config:
            if not PEFT_AVAILABLE:
                raise ImportError("peft is not installed. Please install it to use LoRATrainer with lora_config.")
            
            # Wrap the model with PEFT
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optim=optim,
            epochs=epochs,
            device=device,
            checkpoint_dir=checkpoint_dir,
            checkpoint_epochs=checkpoint_epochs,
            checkpoint_name=checkpoint_name,
            save_best=save_best
        )
