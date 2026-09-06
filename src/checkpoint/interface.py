from typing import Any, Protocol

from .manifest import CheckpointManifest

class CheckpointBackend(Protocol):
    """Abstract interface for checkpoint storage."""
    
    def save(self, context: Any, path: str) -> CheckpointManifest:
        """Save the workload context to the specified path and return a manifest."""
        ...
        
    def load(self, context: Any, path: str) -> CheckpointManifest:
        """Load the workload context from the specified path and return the manifest."""
        ...
