import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class CheckpointManifest:
    """Metadata describing a checkpoint."""
    format_version: int
    experiment_id: str
    run_id: str
    global_step: int
    world_size: int
    rank: int
    backend: str
    seed: int
    workload: str
    state_items: List[str]
    created_at: str
    git_sha: Optional[str] = None
    config_hash: Optional[str] = None

    def save(self, path: str):
        """Save the manifest to a JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
            
    @classmethod
    def load(cls, path: str) -> 'CheckpointManifest':
        """Load a manifest from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
