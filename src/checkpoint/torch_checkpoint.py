import os
import torch
import datetime
from typing import Any, Dict

from .interface import CheckpointBackend
from .manifest import CheckpointManifest
from ..runtime import distributed as dist

class TorchCheckpointBackend(CheckpointBackend):
    """
    A checkpoint backend using standard torch.save and torch.load.
    The context is expected to expose a state_dict() method and a load_state_dict() method,
    or we can assume the context is a dict of stateful objects.
    For this project, context is expected to be a dict:
    {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "ema": ema,
        "global_step": 80,
        ...
    }
    """
    
    def __init__(self, experiment_id: str, run_id: str, workload_name: str, seed: int):
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.workload_name = workload_name
        self.seed = seed

    def save(self, context: Dict[str, Any], path: str) -> CheckpointManifest:
        os.makedirs(path, exist_ok=True)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # Don't import at module level to avoid circular imports if any
        from ..runtime.environment import collect_environment
        
        state_items = []
        save_dict = {}
        
        # We should NOT checkpoint execution metadata like 'rank' and 'world_size'
        # into the payload. The payload should only be true logical state.
        exclude_keys = {"rank", "world_size"}
        
        for key, obj in context.items():
            if key in exclude_keys:
                continue
                
            state_items.append(key)
            if hasattr(obj, "state_dict"):
                save_dict[key] = obj.state_dict()
            else:
                save_dict[key] = obj
                
        torch.save(save_dict, os.path.join(path, f"state_rank{rank}.pt"))
        
        env_meta = collect_environment()
        
        manifest = CheckpointManifest(
            format_version=1,
            experiment_id=self.experiment_id,
            run_id=self.run_id,
            global_step=context.get("global_step", 0),
            world_size=world_size,
            rank=rank,
            backend=dist.dist.get_backend() if dist.is_initialized() else "unknown",
            seed=self.seed,
            workload=self.workload_name,
            state_items=state_items,
            created_at=datetime.datetime.utcnow().isoformat() + "Z",
            git_sha=env_meta.get("git_sha"),
            config_hash=None,
            environment=env_meta
        )
        
        manifest.save(os.path.join(path, f"manifest_rank{rank}.json"))
        return manifest
        
    def load(self, context: Dict[str, Any], path: str) -> CheckpointManifest:
        rank = dist.get_rank()
        
        manifest = CheckpointManifest.load(os.path.join(path, f"manifest_rank{rank}.json"))
        
        # In PyTorch 2.1+, weights_only=True is default and safe, but for general Python state
        # like step counters we might need weights_only=False or to load them securely.
        save_dict = torch.load(os.path.join(path, f"state_rank{rank}.pt"), weights_only=False)
        
        for key, saved_val in save_dict.items():
            if key in context:
                obj = context[key]
                if hasattr(obj, "load_state_dict"):
                    obj.load_state_dict(saved_val)
                else:
                    # For scalar values like global_step, we just re-assign in the dict.
                    # This modifies the context dictionary directly.
                    context[key] = saved_val
                    
        return manifest
