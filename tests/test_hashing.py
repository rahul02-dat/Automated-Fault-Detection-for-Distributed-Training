"""
Tests for hash determinism (P11).

Covers: identical state → identical digest, dtype/shape/key/content changes
produce different digests, canonicalization order independence.
"""
import pytest
import torch
import copy

from src.validator.hashing import compute_digest
from src.validator.comparison import canonicalize


class TestHashDeterminism:
    """Identical state must produce identical digests."""

    def test_same_tensor_same_digest(self):
        t = torch.randn(4, 4)
        d1 = compute_digest(t)
        d2 = compute_digest(t.clone())
        assert d1 == d2

    def test_same_scalar_same_digest(self):
        assert compute_digest(42) == compute_digest(42)

    def test_same_dict_same_digest(self):
        d1 = {"a": torch.ones(2), "b": 3}
        d2 = {"a": torch.ones(2), "b": 3}
        assert compute_digest(canonicalize(d1)) == compute_digest(canonicalize(d2))

    def test_same_list_same_digest(self):
        l1 = [1, 2, torch.ones(3)]
        l2 = [1, 2, torch.ones(3)]
        assert compute_digest(l1) == compute_digest(l2)

    def test_empty_tensor_same_digest(self):
        t1 = torch.tensor([])
        t2 = torch.tensor([])
        assert compute_digest(t1) == compute_digest(t2)


class TestHashSensitivity:
    """Changes in dtype, shape, key, or content must change the digest."""

    def test_content_change(self):
        t1 = torch.ones(3)
        t2 = torch.zeros(3)
        assert compute_digest(t1) != compute_digest(t2)

    def test_shape_change(self):
        t1 = torch.ones(6)
        t2 = torch.ones(2, 3)
        assert compute_digest(t1) != compute_digest(t2)

    def test_dtype_change(self):
        t1 = torch.ones(3, dtype=torch.float32)
        t2 = torch.ones(3, dtype=torch.float64)
        assert compute_digest(t1) != compute_digest(t2)

    def test_dict_key_change(self):
        d1 = canonicalize({"a": 1})
        d2 = canonicalize({"b": 1})
        assert compute_digest(d1) != compute_digest(d2)

    def test_dict_value_change(self):
        d1 = canonicalize({"a": 1})
        d2 = canonicalize({"a": 2})
        assert compute_digest(d1) != compute_digest(d2)

    def test_list_order_matters(self):
        l1 = [1, 2, 3]
        l2 = [3, 2, 1]
        assert compute_digest(l1) != compute_digest(l2)

    def test_scalar_type_matters(self):
        assert compute_digest(1) != compute_digest(1.0)


class TestDictOrderIndependence:
    """Dict insertion order must not affect the digest (after canonicalization)."""

    def test_insertion_order_independent(self):
        d1 = {"z": 1, "a": 2, "m": 3}
        d2 = {"a": 2, "m": 3, "z": 1}
        c1 = canonicalize(d1)
        c2 = canonicalize(d2)
        assert compute_digest(c1) == compute_digest(c2)

    def test_nested_dict_order_independent(self):
        d1 = {"outer": {"z": 1, "a": 2}}
        d2 = {"outer": {"a": 2, "z": 1}}
        c1 = canonicalize(d1)
        c2 = canonicalize(d2)
        assert compute_digest(c1) == compute_digest(c2)
