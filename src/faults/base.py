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
    def apply(self, state: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Applies the fault to the given state dict or context.
        Returns the mutated state.
        """
        pass

    @abc.abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        Returns metadata describing the fault injection properties, 
        e.g., target, scope, deterministic flag.
        """
        pass

