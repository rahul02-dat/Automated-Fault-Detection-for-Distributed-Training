import abc
from typing import Any, Dict, Optional


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

    @property
    @abc.abstractmethod
    def target_state(self) -> str:
        """The logical name of the state this fault targets (e.g. 'ema.step', 'scheduler')."""
        pass

    @abc.abstractmethod
    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the fault directly to the loaded checkpoint payload dict.
        Returns the mutated state payload.
        """
        pass

    @abc.abstractmethod
    def verify_mutation(self, before_state: Dict[str, Any], after_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify that the intended mutation was actually applied.

        Returns a report dict containing at minimum:
            fault: str
            state: str (target state path)
            mutation_applied: bool
            before: <value or digest>
            after: <value or digest>

        A fault experiment is invalid if the intended state was not changed.
        """
        pass

    @abc.abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        Returns metadata describing the fault injection properties,
        e.g., target, scope, deterministic flag.
        Must include a 'fault' key with the fault name.
        """
        pass
