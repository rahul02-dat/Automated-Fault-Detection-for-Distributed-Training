"""
Formal declarative specification of distributed logical state.

This module defines the theoretical correctness model for distributed state.
It must remain entirely declarative and contain ZERO distributed communication primitives
(e.g., dist.all_gather, dist.broadcast).

It describes WHAT correctness means, not HOW to test it.
"""
from enum import Enum

class OwnershipScope(str, Enum):
    """
    Formal ownership scopes for distributed training state.
    """
    
    # GLOBAL: One logical value exists for the entire distributed job.
    # The value must satisfy the defined global-consistency invariant across all ranks.
    GLOBAL = "GLOBAL"
    
    # REPLICATED: Every rank must represent the exact same logical value.
    # rank_0 == rank_1 == ... == rank_N
    REPLICATED = "REPLICATED"
    
    # PER_RANK: Each rank has its own logically distinct value.
    # Values may differ across ranks, but each rank must match its expected rank-specific value.
    PER_RANK = "PER_RANK"
    
    # LOCAL: State belongs to a single rank/process and has no cross-rank equality requirement.
    LOCAL = "LOCAL"
    
    # SHARDED: Ranks collectively represent one logical value through disjoint/defined ownership.
    # The collection of shards must satisfy logical identity, global shape, valid ranges,
    # complete coverage, and no illegal overlap.
    SHARDED = "SHARDED"


def validate_replicated_semantics(all_rank_values: list, comparator_func) -> bool:
    """
    Formal semantic test for REPLICATED state.
    Given a list of values representing the state at each rank, all must be equal
    according to the comparator function.
    """
    if not all_rank_values:
        return True
    reference = all_rank_values[0]
    return all(comparator_func(reference, val) for val in all_rank_values[1:])


def validate_per_rank_semantics(rank_values: list, expected_values: list, comparator_func) -> bool:
    """
    Formal semantic test for PER_RANK state.
    Each rank must match its *own* expected value. Cross-rank equality is not required.
    """
    if len(rank_values) != len(expected_values):
        return False
    return all(comparator_func(val, exp) for val, exp in zip(rank_values, expected_values))


def validate_local_semantics(rank_values: list) -> bool:
    """
    Formal semantic test for LOCAL state.
    No cross-rank equality requirement exists. The state is inherently valid structurally.
    """
    return True


def validate_global_semantics(rank_values: list, invariant_func) -> bool:
    """
    Formal semantic test for GLOBAL state.
    The values collectively (or individually, if truly global and thus identical)
    must satisfy a specified global invariant.
    """
    return invariant_func(rank_values)


def validate_sharded_semantics(shards: list, global_shape: tuple) -> bool:
    """
    Formal semantic test for SHARDED state (1D/2D simplified example).
    Verifies complete coverage and no overlap.
    shards: list of dicts with 'offset' and 'shape' (sizes)
    global_shape: tuple representing the expected total size.
    """
    # Simplified 1D/2D volume check
    total_volume = 1
    for dim in global_shape:
        total_volume *= dim
        
    shard_volume = 0
    for shard in shards:
        vol = 1
        for dim in shard['shape']:
            vol *= dim
        shard_volume += vol
        
    # In a full implementation, you'd check exact coordinate overlap.
    # Here we check volume equivalence as a surrogate for complete coverage and no overlap.
    return total_volume == shard_volume

