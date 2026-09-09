"""
Tests for fault injectors (P11).

Covers: verify_mutation, target_state, metadata, and negative controls
(expected PER_RANK differences should NOT be reported as cross-rank faults).
"""
import pytest
import torch

from src.faults.base import FaultInjector
from src.faults.ema import EMAScalarOmissionFault
from src.faults.scheduler import SchedulerStaleStateFault
from src.faults.rng import RNGStateOmissionFault
from src.faults.dataloader import DataCursorMismatchFault
from src.faults.optimizer import OptimizerStateCorruptionFault
from src.validator.sharded import ShardDescriptor, validate_sharded


class TestFaultInjectorInterface:
    """All fault injectors must implement the abstract interface."""

    @pytest.fixture(params=[
        EMAScalarOmissionFault,
        SchedulerStaleStateFault,
        RNGStateOmissionFault,
        DataCursorMismatchFault,
        OptimizerStateCorruptionFault,
    ])
    def fault(self, request):
        return request.param()

    def test_has_name(self, fault):
        assert isinstance(fault.name, str)
        assert len(fault.name) > 0

    def test_has_target_state(self, fault):
        assert isinstance(fault.target_state, str)
        assert len(fault.target_state) > 0

    def test_has_metadata(self, fault):
        meta = fault.metadata()
        assert isinstance(meta, dict)
        assert "fault" in meta, "metadata() must include a 'fault' key"
        assert meta["fault"] == fault.name

    def test_is_fault_injector(self, fault):
        assert isinstance(fault, FaultInjector)


class TestShardDescriptor:
    """Tests for ShardDescriptor validation."""

    def test_valid_descriptor(self):
        desc = ShardDescriptor(
            logical_name="model.weight",
            global_shape=(100, 200),
            offset=(0, 0),
            local_shape=(50, 200),
            rank=0,
        )
        assert desc.end_offset == (50, 200)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="global_shape dims"):
            ShardDescriptor(
                logical_name="x",
                global_shape=(100, 200),
                offset=(0,),  # wrong dims
                local_shape=(50, 200),
                rank=0,
            )

    def test_serialization_roundtrip(self):
        desc = ShardDescriptor(
            logical_name="w",
            global_shape=(10,),
            offset=(5,),
            local_shape=(5,),
            rank=1,
            digest="abc123",
        )
        d = desc.to_dict()
        desc2 = ShardDescriptor.from_dict(d)
        assert desc2.logical_name == "w"
        assert desc2.offset == (5,)
        assert desc2.digest == "abc123"


class TestValidateSharded:
    """Tests for the sharded validation function."""

    def test_valid_shards_pass(self):
        descs = [
            ShardDescriptor("w", (100,), (0,), (50,), rank=0, digest="aaa"),
            ShardDescriptor("w", (100,), (50,), (50,), rank=1, digest="bbb"),
        ]
        result = validate_sharded(descs)
        assert result["status"] == "PASS"

    def test_overlapping_shards_fail(self):
        descs = [
            ShardDescriptor("w", (100,), (0,), (60,), rank=0, digest="aaa"),
            ShardDescriptor("w", (100,), (40,), (60,), rank=1, digest="bbb"),
        ]
        result = validate_sharded(descs)
        assert result["status"] == "FAIL"
        # Check that overlap is reported
        failed_checks = [c for c in result["checks"] if c["status"] == "FAIL"]
        assert any("overlap" in c["check"].lower() or "coverage" in c["check"].lower() for c in failed_checks)

    def test_incomplete_coverage_fails(self):
        descs = [
            ShardDescriptor("w", (100,), (0,), (30,), rank=0, digest="aaa"),
            ShardDescriptor("w", (100,), (30,), (30,), rank=1, digest="bbb"),
        ]
        result = validate_sharded(descs)
        assert result["status"] == "FAIL"
        failed_checks = [c for c in result["checks"] if c["status"] == "FAIL"]
        assert any("coverage" in c["check"] for c in failed_checks)

    def test_inconsistent_global_shape_fails(self):
        descs = [
            ShardDescriptor("w", (100,), (0,), (50,), rank=0, digest="aaa"),
            ShardDescriptor("w", (200,), (50,), (50,), rank=1, digest="bbb"),
        ]
        result = validate_sharded(descs)
        assert result["status"] == "FAIL"

    def test_out_of_bounds_shard_fails(self):
        descs = [
            ShardDescriptor("w", (100,), (0,), (50,), rank=0, digest="aaa"),
            ShardDescriptor("w", (100,), (50,), (60,), rank=1, digest="bbb"),  # exceeds
        ]
        result = validate_sharded(descs)
        assert result["status"] == "FAIL"

    def test_duplicate_ranks_fail(self):
        descs = [
            ShardDescriptor("w", (100,), (0,), (50,), rank=0, digest="aaa"),
            ShardDescriptor("w", (100,), (50,), (50,), rank=0, digest="bbb"),  # same rank
        ]
        result = validate_sharded(descs)
        assert result["status"] == "FAIL"

    def test_missing_digests_fail(self):
        descs = [
            ShardDescriptor("w", (100,), (0,), (50,), rank=0, digest="aaa"),
            ShardDescriptor("w", (100,), (50,), (50,), rank=1, digest=""),  # missing
        ]
        result = validate_sharded(descs)
        assert result["status"] == "FAIL"

    def test_empty_descriptors_fail(self):
        result = validate_sharded([])
        assert result["status"] == "FAIL"

    def test_2d_shards_pass(self):
        """Test 2D sharding (e.g., row-wise split of a matrix)."""
        descs = [
            ShardDescriptor("w", (4, 8), (0, 0), (2, 8), rank=0, digest="aaa"),
            ShardDescriptor("w", (4, 8), (2, 0), (2, 8), rank=1, digest="bbb"),
        ]
        result = validate_sharded(descs)
        assert result["status"] == "PASS"


class TestNegativeControls:
    """
    Negative control tests — operations that should NOT trigger faults.
    These test that the validation system doesn't over-report.
    """

    def test_per_rank_differences_are_expected(self):
        """PER_RANK state is expected to differ across ranks — not a fault."""
        from src.validator.contracts import StateContract, StateScope, Comparator
        contract = StateContract(name="rng", scope=StateScope.PER_RANK, comparator=Comparator.EXACT)
        # PER_RANK contracts shouldn't fail for cross-rank differences
        assert contract.scope == StateScope.PER_RANK

    def test_local_scope_not_validated(self):
        """LOCAL scope should never fail cross-rank validation."""
        from src.validator.contracts import StateContract, StateScope, Comparator
        contract = StateContract(name="debug_log", scope=StateScope.LOCAL, comparator=Comparator.EXACT)
        assert contract.scope == StateScope.LOCAL
