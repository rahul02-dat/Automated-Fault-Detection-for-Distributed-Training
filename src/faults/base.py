import abc
from typing import Any, Dict

class FaultInjector(abc.ABC):
    """
    Abstract base class for all fault injectors.
    A fault injector artificially corrupts or alters training state,
    usually at a checkpoint restore boundary, to simulate silent bugs.
    """
    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the fault directly to the loaded checkpoint payload dict.
        Returns the mutated state payload.
        """
        pass

    @abc.abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        Returns metadata describing the fault injection properties, 
        e.g., target, scope, deterministic flag.
        """
        pass

