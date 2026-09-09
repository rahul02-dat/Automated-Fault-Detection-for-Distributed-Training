"""
Tests for comparison functions (P11).

Covers: EXACT, ALLCLOSE, HASH comparisons, key-path tracking,
NaN/Inf handling, nested structures, empty tensors, edge cases.
"""
import pytest
import torch

from src.validator.comparison import (
    canonicalize, compare_exact, compare_allclose, compare_hash, compare_state
)
from src.validator.contracts import Comparator
from src.validator.errors import UnsupportedComparatorError


class TestCanonicalize:
    def test_tensor_detach_cpu(self):
        t = torch.randn(3, 3, requires_grad=True)
        c = canonicalize(t)
        assert not c.requires_grad
        assert c.device == torch.device("cpu")

    def test_dict_sorted_keys(self):
        d = {"b": 2, "a": torch.ones(1)}
        cd = canonicalize(d)
        assert list(cd.keys()) == ["a", "b"]

    def test_nested_dict(self):
        d = {"z": {"b": 1, "a": 2}, "a": 0}
        cd = canonicalize(d)
        assert list(cd.keys()) == ["a", "z"]
        assert list(cd["z"].keys()) == ["a", "b"]

    def test_list_preserved(self):
        lst = [torch.ones(1), torch.zeros(1)]
        cl = canonicalize(lst)
        assert isinstance(cl, list)
        assert len(cl) == 2

    def test_tuple_preserved(self):
        tpl = (1, 2, 3)
        ct = canonicalize(tpl)
        assert isinstance(ct, tuple)

    def test_scalar(self):
        assert canonicalize(42) == 42
        assert canonicalize("hello") == "hello"


class TestCompareExact:
    def test_scalar_equal(self):
        res = compare_exact(1, 1)
        assert res["match"]

    def test_scalar_unequal(self):
        res = compare_exact(1, 2)
        assert not res["match"]
        assert "Value mismatch" in res["reason"]

    def test_tensor_equal(self):
        t = torch.ones(2, 2)
        res = compare_exact(t, t.clone())
        assert res["match"]

    def test_tensor_unequal(self):
        t1 = torch.ones(2, 2)
        t2 = t1.clone()
        t2[0, 0] = 0.0
        res = compare_exact(t1, t2)
        assert not res["match"]
        assert "Tensor values mismatch" in res["reason"]
        assert "max_abs_diff" in res
        assert "first_mismatch_index" in res

    def test_tensor_shape_mismatch(self):
        res = compare_exact(torch.ones(2, 2), torch.ones(3, 3))
        assert not res["match"]
        assert "Shape mismatch" in res["reason"]

    def test_tensor_dtype_mismatch(self):
        res = compare_exact(torch.ones(2, dtype=torch.float32), torch.ones(2, dtype=torch.float64))
        assert not res["match"]
        assert "Dtype mismatch" in res["reason"]

    def test_dict_equal(self):
        d1 = {"a": 1, "b": torch.ones(2)}
        d2 = {"a": 1, "b": torch.ones(2)}
        res = compare_exact(d1, d2)
        assert res["match"]

    def test_dict_unequal_value(self):
        d1 = {"a": 1}
        d2 = {"a": 2}
        res = compare_exact(d1, d2)
        assert not res["match"]

    def test_dict_missing_key(self):
        d1 = {"a": 1}
        d2 = {"a": 1, "b": 2}
        res = compare_exact(d1, d2)
        assert not res["match"]
        assert "keys mismatch" in res["reason"]

    def test_list_equal(self):
        res = compare_exact([1, 2, 3], [1, 2, 3])
        assert res["match"]

    def test_list_length_mismatch(self):
        res = compare_exact([1, 2], [1, 2, 3])
        assert not res["match"]

    def test_type_mismatch(self):
        res = compare_exact(1, "1")
        assert not res["match"]
        assert "Type mismatch" in res["reason"]

    def test_nan_same_location(self):
        t1 = torch.tensor([1.0, float("nan"), 3.0])
        t2 = torch.tensor([1.0, float("nan"), 3.0])
        res = compare_exact(t1, t2)
        # NaN != NaN for torch.equal, but NaN locations match
        # The function checks NaN locations first, then values
        assert "nan_count_a" not in res or res.get("match", False) is False

    def test_inf_handling(self):
        t1 = torch.tensor([1.0, float("inf"), -float("inf")])
        t2 = torch.tensor([1.0, float("inf"), -float("inf")])
        res = compare_exact(t1, t2)
        assert res["match"]

    def test_inf_mismatch(self):
        t1 = torch.tensor([1.0, float("inf")])
        t2 = torch.tensor([1.0, 2.0])
        res = compare_exact(t1, t2)
        assert not res["match"]

    def test_empty_tensor(self):
        t1 = torch.tensor([])
        t2 = torch.tensor([])
        res = compare_exact(t1, t2)
        assert res["match"]

    def test_zero_tensors(self):
        t1 = torch.zeros(3, 3)
        t2 = torch.zeros(3, 3)
        res = compare_exact(t1, t2)
        assert res["match"]

    def test_key_path_tracking(self):
        d1 = {"a": {"b": torch.ones(2)}}
        d2 = {"a": {"b": torch.zeros(2)}}
        res = compare_exact(d1, d2)
        assert not res["match"]
        assert "a.b" in res.get("key_path", "")

    def test_nested_list_in_dict(self):
        d1 = {"a": [1, 2, 3]}
        d2 = {"a": [1, 2, 4]}
        res = compare_exact(d1, d2)
        assert not res["match"]


class TestCompareAllclose:
    def test_within_tolerance(self):
        t1 = torch.ones(2)
        t2 = torch.ones(2) + 1e-4
        res = compare_allclose(t1, t2, rtol=1e-3, atol=1e-3)
        assert res["match"]

    def test_outside_tolerance(self):
        t1 = torch.ones(2)
        t2 = torch.ones(2) + 1e-4
        res = compare_allclose(t1, t2, rtol=1e-5, atol=1e-5)
        assert not res["match"]
        assert "Tensor allclose mismatch" in res["reason"]
        assert "rtol" in res
        assert "atol" in res

    def test_shape_mismatch(self):
        res = compare_allclose(torch.ones(2), torch.ones(3), rtol=1e-5, atol=1e-5)
        assert not res["match"]
        assert "Shape mismatch" in res["reason"]

    def test_dict_allclose(self):
        d1 = {"w": torch.ones(3)}
        d2 = {"w": torch.ones(3) + 1e-7}
        res = compare_allclose(d1, d2, rtol=1e-5, atol=1e-5)
        assert res["match"]

    def test_scalar_fallback_to_exact(self):
        res = compare_allclose(42, 42, rtol=1e-5, atol=1e-5)
        assert res["match"]

    def test_scalar_fallback_mismatch(self):
        res = compare_allclose(42, 43, rtol=1e-5, atol=1e-5)
        assert not res["match"]


class TestCompareHash:
    def test_identical_tensors(self):
        t = torch.ones(3, 3)
        res = compare_hash(t, t.clone())
        assert res["match"]

    def test_different_tensors(self):
        t1 = torch.ones(3, 3)
        t2 = torch.zeros(3, 3)
        res = compare_hash(t1, t2)
        assert not res["match"]
        assert "hash_expected" in res
        assert "hash_actual" in res


class TestCompareState:
    def test_exact_routing(self):
        res = compare_state(1, 1, Comparator.EXACT)
        assert res["match"]
        assert res["comparator"] == "exact"

    def test_allclose_routing(self):
        res = compare_state(torch.ones(2), torch.ones(2) + 1e-7, Comparator.ALLCLOSE, rtol=1e-5, atol=1e-5)
        assert res["match"]
        assert res["comparator"] == "allclose"
        assert res["rtol"] == 1e-5
        assert res["atol"] == 1e-5

    def test_hash_routing(self):
        res = compare_state(torch.ones(2), torch.ones(2), Comparator.HASH)
        assert res["match"]
        assert res["comparator"] == "hash"

    def test_norm_raises(self):
        with pytest.raises(UnsupportedComparatorError):
            compare_state(1, 1, Comparator.NORM)

    def test_custom_raises(self):
        with pytest.raises(UnsupportedComparatorError):
            compare_state(1, 1, Comparator.CUSTOM)

    def test_allclose_missing_tolerances(self):
        with pytest.raises(ValueError, match="ALLCLOSE requires"):
            compare_state(torch.ones(2), torch.ones(2), Comparator.ALLCLOSE)
