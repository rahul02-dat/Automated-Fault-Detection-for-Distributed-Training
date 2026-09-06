import time
from typing import Any, Dict, List, Optional

from .contracts import StateContract, StateScope, Comparator
from .registry import StateRegistry
from .comparison import canonicalize, compare_state
from ..runtime import distributed as dist


def _validate_global_or_replicated(
    contract: StateContract, 
    local_val: Any, 
    rank: int, 
    world_size: int
) -> Dict[str, Any]:
    """Validates GLOBAL or REPLICATED state across all ranks."""
    # 1. Canonicalize
    canon_val = canonicalize(local_val)
    
    # 2. Gather values to all ranks
    all_vals = dist.gather_object(canon_val)
    
    # 3. Rank 0 computes verdict
    if rank == 0:
        expected = all_vals[0]
        status = "PASS"
        message = None
        max_abs_diff = None
        
        rank_results = []
        for r in range(world_size):
            match, reason = compare_state(
                expected, 
                all_vals[r], 
                contract.comparator, 
                contract.rtol, 
                contract.atol
            )
            rank_results.append({
                "rank": r,
                "match": match,
                "reason": reason
            })
            if not match:
                status = "FAIL"
                if message is None:
                    message = f"Rank {r} diverges from Rank 0: {reason}"
                    
        return {
            "status": status,
            "state_name": contract.name,
            "scope": contract.scope.value,
            "comparator": contract.comparator.value,
            "rank_results": rank_results,
            "message": message
        }
    else:
        return {}


def _validate_per_rank(
    contract: StateContract, 
    local_val: Any, 
    expected_val: Any, 
    rank: int
) -> Dict[str, Any]:
    """Validates PER_RANK state against an expected value (e.g. from reference)."""
    canon_val = canonicalize(local_val)
    canon_expected = canonicalize(expected_val)
    
    match, reason = compare_state(
        canon_expected, 
        canon_val, 
        contract.comparator, 
        contract.rtol, 
        contract.atol
    )
    
    local_result = {
        "status": "PASS" if match else "FAIL",
        "rank": rank,
        "reason": reason
    }
    
    all_results = dist.gather_object(local_result)
    
    if rank == 0:
        status = "PASS"
        message = None
        for res in all_results:
            if res["status"] == "FAIL":
                status = "FAIL"
                if message is None:
                    message = f"Rank {res['rank']} mismatch: {res['reason']}"
                    
        return {
            "status": status,
            "state_name": contract.name,
            "scope": contract.scope.value,
            "comparator": contract.comparator.value,
            "rank_results": all_results,
            "message": message
        }
    else:
        return {}


def validate_cross_rank(registry: StateRegistry, context: Any) -> List[Dict[str, Any]]:
    """
    Validates cross-rank consistency for all registered contracts.
    Only GLOBAL and REPLICATED scopes are validated here.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    results = []
    
    for name, contract in registry.get_all_contracts().items():
        start_t = time.time()
        try:
            local_val = registry.extract_state(name, context)
            
            if contract.scope in (StateScope.GLOBAL, StateScope.REPLICATED):
                result = _validate_global_or_replicated(contract, local_val, rank, world_size)
            elif contract.scope == StateScope.SHARDED:
                # Sharded logic not fully implemented yet per GUIDELINES
                result = {
                    "status": "SKIP",
                    "state_name": contract.name,
                    "scope": contract.scope.value,
                    "message": "Sharded cross-rank validation not yet implemented."
                }
            else:
                # LOCAL or PER_RANK shouldn't be blindly compared cross-rank
                result = {
                    "status": "SKIP",
                    "state_name": contract.name,
                    "scope": contract.scope.value,
                    "message": "Scope not applicable for basic cross-rank equality."
                }
                
        except Exception as e:
            result = {
                "status": "UNKNOWN" if not contract.required else "FAIL",
                "state_name": contract.name,
                "scope": contract.scope.value,
                "message": f"Exception during validation: {str(e)}"
            }
            
        if rank == 0 and result:
            result["duration_ms"] = (time.time() - start_t) * 1000
            results.append(result)
            
    return results
