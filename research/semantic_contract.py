"""
Semantic Contracts for Distributed Training State.

This module defines concrete assertions about specific pieces of training state,
mapping them to the formal semantic scopes defined in state_model.py.

It separates the theoretical meaning (state_model.py) from the concrete assertion
(semantic_contract.py), allowing different training strategies (e.g. DDP vs FSDP)
to use different contracts for the same state components (e.g. optimizer).
"""
from dataclasses import dataclass, field
from typing import List

from .state_model import OwnershipScope


@dataclass
class StateContract:
    """
    A concrete assertion about a specific piece of training state.
    
    Example:
        StateContract(name="model", scope=OwnershipScope.REPLICATED, comparator="ALLCLOSE")
        means the logical model state is replicated, and replicas are compared using ALLCLOSE.
    """
    name: str
    scope: OwnershipScope
    comparator: str = "EXACT"

    def __post_init__(self):
        valid_scopes = {e.value for e in OwnershipScope}
        # Assuming the scope could be passed as string or enum
        scope_val = self.scope.value if isinstance(self.scope, OwnershipScope) else self.scope
        if scope_val not in valid_scopes:
            raise ValueError(f"Invalid scope: {self.scope}")


@dataclass
class ContractProfile:
    """
    A profile mapping a specific workload or distributed strategy to its
    expected semantic contracts.
    
    Example:
        ContractProfile(
            workload="resnet_ddp",
            contracts=[
                StateContract(name="model", scope=OwnershipScope.REPLICATED, comparator="ALLCLOSE"),
                StateContract(name="optimizer", scope=OwnershipScope.REPLICATED, comparator="ALLCLOSE"),
            ]
        )
    """
    workload: str
    contracts: List[StateContract] = field(default_factory=list)
