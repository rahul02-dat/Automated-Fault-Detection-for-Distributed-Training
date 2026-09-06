class ValidatorError(Exception):
    """Base class for all validator-related errors."""
    pass

class InvalidContractError(ValidatorError):
    """Raised when a state contract definition is invalid."""
    pass

class UnsupportedScopeError(ValidatorError):
    """Raised when an unsupported StateScope is used."""
    pass

class UnsupportedComparatorError(ValidatorError):
    """Raised when an unsupported Comparator is used."""
    pass

class StateAccessError(ValidatorError):
    """Raised when state cannot be extracted via the provided getter."""
    pass
