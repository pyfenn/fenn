from dataclasses import dataclass
from typing import Tuple


@dataclass
class LoRAConfig:
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
