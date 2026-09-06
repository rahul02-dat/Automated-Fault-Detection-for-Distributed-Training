import random
import numpy as np
import torch
from typing import Optional

def set_deterministic_seeds(global_seed: int, rank: int, local_rank: Optional[int] = None):
    """
    Establish deterministic seeds for the experiment.
    As per GUIDELINES.md:
    global_seed = S
    rank_seed = S + rank
    """
    rank_seed = global_seed + rank
    
    # Python RNG
    random.seed(rank_seed)
    
    # NumPy RNG
    np.random.seed(rank_seed)
    
    # PyTorch CPU RNG
    torch.manual_seed(rank_seed)
    
    # PyTorch CUDA RNG if applicable
    if torch.cuda.is_available():
        if local_rank is not None:
            torch.cuda.set_device(local_rank)
        torch.cuda.manual_seed(rank_seed)
        torch.cuda.manual_seed_all(rank_seed)
        
    # Set deterministic algorithms if required for absolute strictness, 
    # though this can hurt performance or crash on some operations.
    # We will leave these commented out unless strictly needed.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
