"""
Tests for StateContract and StateScope validation (P11).

Covers: scope+comparator combinations, tolerance validation,
unsupported comparator early-fail, and contract properties.
"""
import pytest
from src.validator.contracts import StateContract, StateScope, Comparator
from src.validator.errors import UnsupportedComparatorError


class TestStateContract:
    """Tests for the StateContract dataclass."""

    def test_exact_contract_creation(self):
        c = StateContract(name="step", scope=StateScope.GLOBAL, comparator=Comparator.EXACT)
        assert c.name == "step"
        assert c.scope == StateScope.GLOBAL
        assert c.comparator == Comparator.EXACT
        assert c.required is True

    def test_allclose_requires_tolerances(self):
        with pytest.raises(ValueError, match="rtol or atol"):
            StateContract(name="model", scope=StateScope.REPLICATED, comparator=Comparator.ALLCLOSE)

    def test_allclose_missing_atol_raises(self):
        with pytest.raises(ValueError, match="rtol or atol"):
            StateContract(name="model", scope=StateScope.REPLICATED, comparator=Comparator.ALLCLOSE, rtol=1e-5)

    def test_allclose_missing_rtol_raises(self):
        with pytest.raises(ValueError, match="rtol or atol"):
            StateContract(name="model", scope=StateScope.REPLICATED, comparator=Comparator.ALLCLOSE, atol=1e-5)

    def test_allclose_with_tolerances(self):
        c = StateContract(
            name="model", scope=StateScope.REPLICATED,
            comparator=Comparator.ALLCLOSE, rtol=1e-5, atol=1e-8
        )
        assert c.rtol == 1e-5
        assert c.atol == 1e-8

    def test_hash_contract(self):
        c = StateContract(name="rng", scope=StateScope.PER_RANK, comparator=Comparator.HASH)
        assert c.comparator == Comparator.HASH

    def test_norm_comparator_raises(self):
        with pytest.raises(UnsupportedComparatorError, match="not implemented"):
            StateContract(name="x", scope=StateScope.GLOBAL, comparator=Comparator.NORM)

    def test_custom_comparator_raises(self):
        with pytest.raises(UnsupportedComparatorError, match="not implemented"):
            StateContract(name="x", scope=StateScope.GLOBAL, comparator=Comparator.CUSTOM)

    def test_optional_contract(self):
        c = StateContract(
            name="optional_state", scope=StateScope.LOCAL,
            comparator=Comparator.EXACT, required=False
        )
        assert c.required is False

    def test_contract_description(self):
        c = StateContract(
            name="step", scope=StateScope.GLOBAL,
            comparator=Comparator.EXACT, description="Global training step"
        )
        assert c.description == "Global training step"

    def test_contract_is_frozen(self):
        c = StateContract(name="step", scope=StateScope.GLOBAL, comparator=Comparator.EXACT)
        with pytest.raises(AttributeError):
            c.name = "other"


class TestStateScope:
    """Tests for StateScope enum values."""

    def test_all_scopes_exist(self):
        assert StateScope.GLOBAL.value == "global"
        assert StateScope.REPLICATED.value == "replicated"
        assert StateScope.SHARDED.value == "sharded"
        assert StateScope.PER_RANK.value == "per_rank"
        assert StateScope.LOCAL.value == "local"


class TestComparator:
    """Tests for Comparator enum and is_supported property."""

    def test_supported_comparators(self):
        assert Comparator.EXACT.is_supported is True
        assert Comparator.ALLCLOSE.is_supported is True
        assert Comparator.HASH.is_supported is True

    def test_unsupported_comparators(self):
        assert Comparator.NORM.is_supported is False
        assert Comparator.CUSTOM.is_supported is False

    def test_all_scope_comparator_combinations(self):
        """Test that all valid scope+supported comparator combinations can be created."""
        for scope in [StateScope.GLOBAL, StateScope.REPLICATED, StateScope.PER_RANK, StateScope.SHARDED, StateScope.LOCAL]:
            # EXACT
            StateContract(name=f"test_{scope.value}_exact", scope=scope, comparator=Comparator.EXACT)
            # HASH
            StateContract(name=f"test_{scope.value}_hash", scope=scope, comparator=Comparator.HASH)
            # ALLCLOSE (with tolerances)
            StateContract(
                name=f"test_{scope.value}_allclose", scope=scope,
                comparator=Comparator.ALLCLOSE, rtol=1e-5, atol=1e-8
            )
