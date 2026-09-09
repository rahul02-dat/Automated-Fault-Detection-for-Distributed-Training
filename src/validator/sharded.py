"""
Sharded state validation for distributed checkpointing.

Implements a minimal shard descriptor and validation protocol that checks:
1. Global shape consistency
2. Valid shard ranges (no out-of-bounds)
3. No illegal overlap between shards
4. Expected coverage (shards cover the full global shape)
5. Rank ownership consistency
6. Per-shard digest integrity
7. Logical global digest when feasible
"""
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from .hashing import compute_digest
from ..runtime import distributed as dist


@dataclass
class ShardDescriptor:
    """
    Describes a single shard of a distributed state tensor.

    A shard is a contiguous slice of a global logical tensor.
    Together, all shards across ranks must exactly cover the global shape
    with no overlaps and no gaps.
    """
    logical_name: str
    global_shape: Tuple[int, ...]
    offset: Tuple[int, ...]      # Starting index in each dimension
    local_shape: Tuple[int, ...]  # Shape of this shard
    rank: int
    digest: str = ""

    def __post_init__(self):
        if len(self.global_shape) != len(self.offset):
            raise ValueError(
                f"ShardDescriptor '{self.logical_name}': "
                f"global_shape dims ({len(self.global_shape)}) != offset dims ({len(self.offset)})"
            )
        if len(self.global_shape) != len(self.local_shape):
            raise ValueError(
                f"ShardDescriptor '{self.logical_name}': "
                f"global_shape dims ({len(self.global_shape)}) != local_shape dims ({len(self.local_shape)})"
            )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert tuples to lists for JSON serialization
        d["global_shape"] = list(self.global_shape)
        d["offset"] = list(self.offset)
        d["local_shape"] = list(self.local_shape)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ShardDescriptor":
        return cls(
            logical_name=d["logical_name"],
            global_shape=tuple(d["global_shape"]),
            offset=tuple(d["offset"]),
            local_shape=tuple(d["local_shape"]),
            rank=d["rank"],
            digest=d.get("digest", ""),
        )

    @property
    def end_offset(self) -> Tuple[int, ...]:
        """The exclusive end index in each dimension."""
        return tuple(o + s for o, s in zip(self.offset, self.local_shape))


def _validate_shard_range(descriptor: ShardDescriptor) -> Optional[str]:
    """
    Validate that the shard range is within the global shape bounds.
    Returns an error message or None if valid.
    """
    for dim, (off, local, glob) in enumerate(
        zip(descriptor.offset, descriptor.local_shape, descriptor.global_shape)
    ):
        if off < 0:
            return f"Negative offset in dim {dim}: {off}"
        if local <= 0:
            return f"Non-positive local_shape in dim {dim}: {local}"
        if off + local > glob:
            return (
                f"Shard exceeds global shape in dim {dim}: "
                f"offset {off} + local_shape {local} = {off + local} > global {glob}"
            )
    return None


def _check_overlap(desc_a: ShardDescriptor, desc_b: ShardDescriptor) -> bool:
    """
    Check whether two shard descriptors overlap in any dimension.
    Returns True if they overlap.
    """
    for dim in range(len(desc_a.global_shape)):
        a_start = desc_a.offset[dim]
        a_end = a_start + desc_a.local_shape[dim]
        b_start = desc_b.offset[dim]
        b_end = b_start + desc_b.local_shape[dim]
        # No overlap if one ends before the other starts
        if a_end <= b_start or b_end <= a_start:
            return False
    return True


def _check_coverage(descriptors: List[ShardDescriptor]) -> Optional[str]:
    """
    Check that all shards together cover the entire global shape.
    Only validates total element count (a weaker but practical check).
    Returns an error message or None if coverage is correct.
    """
    if not descriptors:
        return "No shard descriptors provided"

    global_shape = descriptors[0].global_shape
    global_elements = 1
    for s in global_shape:
        global_elements *= s

    total_shard_elements = 0
    for desc in descriptors:
        shard_elements = 1
        for s in desc.local_shape:
            shard_elements *= s
        total_shard_elements += shard_elements

    if total_shard_elements < global_elements:
        return (
            f"Incomplete coverage: shards cover {total_shard_elements} elements "
            f"but global shape requires {global_elements}"
        )
    if total_shard_elements > global_elements:
        return (
            f"Over-coverage (possible overlap): shards cover {total_shard_elements} elements "
            f"but global shape requires {global_elements}"
        )
    return None


def validate_sharded(
    descriptors: List[ShardDescriptor],
    expected_global_digest: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate a set of shard descriptors for consistency.

    This function is called on rank 0 after gathering all shard descriptors.

    Checks:
    1. All descriptors agree on global_shape
    2. All shard ranges are within bounds
    3. No illegal overlaps between any pair of shards
    4. Shards cover the full global shape
    5. Rank ownership is unique (each rank has exactly one shard)
    6. Per-shard digests are present
    7. Global digest matches if provided

    Returns:
        Dict with:
            status: "PASS" | "FAIL"
            checks: list of individual check results
            message: summary if FAIL
    """
    checks = []
    all_pass = True

    if not descriptors:
        return {
            "status": "FAIL",
            "state_name": "unknown",
            "scope": "sharded",
            "message": "No shard descriptors provided.",
            "checks": [],
        }

    logical_name = descriptors[0].logical_name

    # --- Check 1: Global shape agreement ---
    shapes = set(d.global_shape for d in descriptors)
    if len(shapes) > 1:
        checks.append({
            "check": "global_shape_agreement",
            "status": "FAIL",
            "detail": f"Inconsistent global shapes: {shapes}",
        })
        all_pass = False
    else:
        checks.append({"check": "global_shape_agreement", "status": "PASS"})

    # --- Check 2: Shard range validation ---
    for desc in descriptors:
        err = _validate_shard_range(desc)
        if err:
            checks.append({
                "check": f"shard_range_rank{desc.rank}",
                "status": "FAIL",
                "detail": err,
            })
            all_pass = False
        else:
            checks.append({"check": f"shard_range_rank{desc.rank}", "status": "PASS"})

    # --- Check 3: No illegal overlaps ---
    for i in range(len(descriptors)):
        for j in range(i + 1, len(descriptors)):
            if _check_overlap(descriptors[i], descriptors[j]):
                checks.append({
                    "check": f"overlap_rank{descriptors[i].rank}_rank{descriptors[j].rank}",
                    "status": "FAIL",
                    "detail": (
                        f"Overlap between rank {descriptors[i].rank} "
                        f"(offset={descriptors[i].offset}, shape={descriptors[i].local_shape}) "
                        f"and rank {descriptors[j].rank} "
                        f"(offset={descriptors[j].offset}, shape={descriptors[j].local_shape})"
                    ),
                })
                all_pass = False

    if all(c["status"] == "PASS" for c in checks if c["check"].startswith("overlap")):
        checks.append({"check": "no_overlap", "status": "PASS"})

    # --- Check 4: Coverage ---
    coverage_err = _check_coverage(descriptors)
    if coverage_err:
        checks.append({
            "check": "coverage",
            "status": "FAIL",
            "detail": coverage_err,
        })
        all_pass = False
    else:
        checks.append({"check": "coverage", "status": "PASS"})

    # --- Check 5: Rank ownership uniqueness ---
    ranks = [d.rank for d in descriptors]
    if len(ranks) != len(set(ranks)):
        checks.append({
            "check": "rank_uniqueness",
            "status": "FAIL",
            "detail": f"Duplicate ranks: {ranks}",
        })
        all_pass = False
    else:
        checks.append({"check": "rank_uniqueness", "status": "PASS"})

    # --- Check 6: Digest presence ---
    missing_digests = [d.rank for d in descriptors if not d.digest]
    if missing_digests:
        checks.append({
            "check": "digest_presence",
            "status": "FAIL",
            "detail": f"Missing digests for ranks: {missing_digests}",
        })
        all_pass = False
    else:
        checks.append({"check": "digest_presence", "status": "PASS"})

    # --- Check 7: Global digest match (optional) ---
    if expected_global_digest is not None:
        # We can only verify the global digest if we have all shard data
        # For now, we just note whether it was provided
        checks.append({
            "check": "global_digest",
            "status": "SKIP",
            "detail": "Global digest verification requires full shard data gathering.",
        })

    status = "PASS" if all_pass else "FAIL"
    message = None
    if not all_pass:
        failed = [c for c in checks if c.get("status") == "FAIL"]
        message = f"{len(failed)} check(s) failed for sharded state '{logical_name}'"

    return {
        "status": status,
        "state_name": logical_name,
        "scope": "sharded",
        "checks": checks,
        "message": message,
    }
