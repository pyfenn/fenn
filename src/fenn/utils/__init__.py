import datetime
import random
import numpy as np

try:
    import torch
except ImportError as e:
    raise RuntimeError(
        "Torch is required by fenn. Install it yourself (GPU/CPU) or use 'pip install fenn[torch]'."
    ) from e

def set_seed(seed: int) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import secrets
import random
from datetime import datetime

def generate_session_id() -> str:

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # A curated list of "beautiful" words
    #adjectives = [
    #    "autumn", "hidden", "bitter", "misty", "silent",
    #    "empty", "dry", "dark", "summer", "icy", "delicate",
    #    "quiet", "white", "cool", "spring", "winter", "patient",
    #    "twilight", "dawn", "crimson", "wispy", "weathered",
    #    "blue", "billowing", "broken", "cold", "damp", "falling",
    #    "frosty", "green", "long", "late", "lingering"
    #]

    #nouns = [
    #    "waterfall", "river", "breeze", "moon", "rain",
    #    "wind", "sea", "morning", "snow", "lake", "sunset",
    #    "pine", "shadow", "leaf", "dawn", "glitter", "forest",
    #    "hill", "cloud", "meadow", "sun", "glade", "bird",
    #    "brook", "butterfly", "bush", "dew", "dust", "field",
    #    "fire", "flower", "firefly", "feather", "grass"
    #]

    # Select words
    #adj = random.choice(adjectives)
    #noun = random.choice(nouns)

    # Add a secure hex suffix (2 bytes = 4 hex chars) to ensure uniqueness
    hex_suffix = secrets.token_hex(2) 
    
    return timestamp + "_" + hex_suffix