import time
from typing import Any, Dict, List

from .contracts import StateContract, StateScope
from .registry import StateRegistry
from .comparison import canonicalize, compare_state
from .hashing import compute_digest
from ..runtime import distributed as dist

def _validate_global_or_replicated(
    contract: StateContract, 
    local_val: Any, 
    rank: int, 
    world_size: int
) -> Dict[str, Any]:
    """Validates GLOBAL or REPLICATED state across all ranks using a 2-stage protocol."""
    # 1. Canonicalize
    canon_val = canonicalize(local_val)
    
    # 2. Compute local digest
    local_digest = compute_digest(canon_val)
    
    # 3. Gather digests (all_gather_object returns list to all ranks)
    all_digests = dist.gather_object(local_digest)
    
    expected_digest = all_digests[0]
    has_mismatch = any(d != expected_digest for d in all_digests)
    
    # 4. If mismatch, gather full state for detailed comparison
    if has_mismatch:
        all_vals = dist.gather_object(canon_val)
    else:
        all_vals = None
    
    # 5. Rank 0 computes detailed verdict
    if rank == 0:
        status = "PASS"
        message = None
        rank_results = []
        
        for r in range(world_size):
            if all_digests[r] == expected_digest:
                rank_results.append({
                    "rank": r,
                    "match": True,
                    "reason": None
                })
            else:
                status = "FAIL"
                res = compare_state(
                    all_vals[0], 
                    all_vals[r], 
                    contract.comparator, 
                    contract.rtol, 
                    contract.atol
                )
                res["rank"] = r
                rank_results.append(res)
                
                if message is None:
                    message = f"Rank {r} diverges from Rank 0: {res.get('reason', 'Unknown diff')}"
                    
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
    
    # 1. Compute digests
    val_digest = compute_digest(canon_val)
    expected_digest = compute_digest(canon_expected)
    
    has_mismatch = val_digest != expected_digest
    
    # 2. If mismatch, do detailed comparison locally
    if has_mismatch:
        local_result = compare_state(
            canon_expected, 
            canon_val, 
            contract.comparator, 
            contract.rtol, 
            contract.atol
        )
        local_result["rank"] = rank
    else:
        local_result = {"match": True, "rank": rank, "reason": None}
        
    local_result["status"] = "PASS" if local_result["match"] else "FAIL"
    
    # 3. Gather results
    all_results = dist.gather_object(local_result)
    
    if rank == 0:
        status = "PASS"
        message = None
        for res in all_results:
            if res["status"] == "FAIL":
                status = "FAIL"
                if message is None:
                    message = f"Rank {res['rank']} mismatch: {res.get('reason', 'Unknown diff')}"
                    
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


def validate_cross_rank(registry: StateRegistry, context: Any, expected_state: Any = None) -> List[Dict[str, Any]]:
    """
    Validates cross-rank consistency for all registered contracts.
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
            elif contract.scope == StateScope.PER_RANK:
                if expected_state is not None and name in expected_state:
                    expected_val = expected_state[name]
                    result = _validate_per_rank(contract, local_val, expected_val, rank)
                else:
                    result = {
                        "status": "UNKNOWN" if not contract.required else "FAIL",
                        "state_name": contract.name,
                        "scope": contract.scope.value,
                        "message": "Missing expected_state for PER_RANK validation."
                    }
            elif contract.scope == StateScope.SHARDED:
                # Sharded logic not fully implemented yet per GUIDELINES
                result = {
                    "status": "SKIP",
                    "state_name": contract.name,
                    "scope": contract.scope.value,
                    "message": "Sharded cross-rank validation not yet implemented."
                }
            else:
                # LOCAL
                result = {
                    "status": "SKIP",
                    "state_name": contract.name,
                    "scope": contract.scope.value,
                    "message": "LOCAL scope not validated cross-rank."
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
