from typing import Callable, Any, Dict

from .contracts import StateContract
from .errors import InvalidContractError, StateAccessError


class StateRegistry:
    """
    Maps logical state names to their StateContracts and getter functions.
    The getter function takes a context object (e.g., the workload) and returns
    the state value.
    """

    def __init__(self):
        self._contracts: Dict[str, StateContract] = {}
        self._getters: Dict[str, Callable[[Any], Any]] = {}

    def register(self, contract: StateContract, getter: Callable[[Any], Any]) -> None:
        """Register a new state contract and its getter."""
        if not isinstance(contract, StateContract):
            raise InvalidContractError(f"Expected StateContract, got {type(contract)}")

        if contract.name in self._contracts:
            raise InvalidContractError(f"Contract '{contract.name}' is already registered.")

        self._contracts[contract.name] = contract
        self._getters[contract.name] = getter

    def get_contract(self, name: str) -> StateContract:
        """Retrieve a registered StateContract by name."""
        if name not in self._contracts:
            raise InvalidContractError(f"Contract '{name}' is not registered.")
        return self._contracts[name]

    def get_all_contracts(self) -> Dict[str, StateContract]:
        """Retrieve all registered contracts."""
        return self._contracts.copy()

    def extract_state(self, name: str, context: Any) -> Any:
        """Extract state from the context using the registered getter."""
        contract = self.get_contract(name)
        getter = self._getters[name]
        try:
            return getter(context)
        except Exception as e:
            if contract.required:
                raise StateAccessError(
                    f"Failed to extract required state '{name}': {e}"
                ) from e
            return None
