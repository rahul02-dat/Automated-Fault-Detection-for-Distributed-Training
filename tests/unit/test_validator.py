import pytest
import torch

from src.validator.contracts import StateContract, StateScope, Comparator
from src.validator.registry import StateRegistry
from src.validator.errors import InvalidContractError, StateAccessError
from src.validator.comparison import canonicalize, compare_exact, compare_allclose

def test_state_contract_validation():
    # Should work
    StateContract(name="test", scope=StateScope.GLOBAL, comparator=Comparator.EXACT)
    
    # Missing tolerances for ALLCLOSE should raise ValueError
    with pytest.raises(ValueError, match="rtol or atol"):
        StateContract(name="test", scope=StateScope.GLOBAL, comparator=Comparator.ALLCLOSE)
        
    # Providing tolerances should work
    StateContract(
        name="test", 
        scope=StateScope.GLOBAL, 
        comparator=Comparator.ALLCLOSE,
        rtol=1e-5,
        atol=1e-8
    )

def test_registry():
    registry = StateRegistry()
    contract = StateContract(name="step", scope=StateScope.GLOBAL, comparator=Comparator.EXACT)
    
    registry.register(contract, lambda ctx: ctx["step"])
    
    # Test duplicate registration
    with pytest.raises(InvalidContractError):
        registry.register(contract, lambda ctx: ctx["step"])
        
    # Test extract
    context = {"step": 42}
    assert registry.extract_state("step", context) == 42
    
    # Test required extraction failure
    with pytest.raises(StateAccessError):
        registry.extract_state("step", {})

def test_canonicalize():
    t = torch.randn(3, 3, device="cpu", requires_grad=True)
    c = canonicalize(t)
    assert not c.requires_grad
    
    d = {"b": 2, "a": torch.ones(1)}
    cd = canonicalize(d)
    assert list(cd.keys()) == ["a", "b"]

def test_compare_exact():
    # Scalars
    match, _ = compare_exact(1, 1)
    assert match
    match, _ = compare_exact(1, 2)
    assert not match
    
    # Tensors
    t1 = torch.ones(2)
    t2 = torch.ones(2)
    match, _ = compare_exact(t1, t2)
    assert match
    
    t3 = torch.zeros(2)
    match, _ = compare_exact(t1, t3)
    assert not match
    
    # Dicts
    d1 = {"a": t1, "b": 1}
    d2 = {"a": t2, "b": 1}
    match, _ = compare_exact(d1, d2)
    assert match
    
def test_compare_allclose():
    t1 = torch.ones(2)
    t2 = torch.ones(2) + 1e-4
    
    match, _ = compare_allclose(t1, t2, rtol=1e-3, atol=1e-3)
    assert match
    
    match, _ = compare_allclose(t1, t2, rtol=1e-5, atol=1e-5)
    assert not match
