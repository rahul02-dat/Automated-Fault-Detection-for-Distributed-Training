from research.state_model import (
    validate_replicated_semantics,
    validate_per_rank_semantics,
    validate_local_semantics,
    validate_global_semantics,
    validate_sharded_semantics
)

def test_replicated_semantics():
    def exact_cmp(a, b): return a == b
    
    # rank0 == rank1 == rank2 -> valid
    assert validate_replicated_semantics([1, 1, 1], exact_cmp) is True
    
    # rank0 != rank1 -> invalid
    assert validate_replicated_semantics([1, 2, 1], exact_cmp) is False


def test_per_rank_semantics():
    def exact_cmp(a, b): return a == b
    
    # rank0 != rank1 -> valid (provided they match expectations)
    rank_values = [10, 20]
    expected_values = [10, 20]
    assert validate_per_rank_semantics(rank_values, expected_values, exact_cmp) is True
    
    # mismatch expectation -> invalid
    assert validate_per_rank_semantics(rank_values, [10, 30], exact_cmp) is False


def test_local_semantics():
    # rank0 != rank1 -> valid
    assert validate_local_semantics([10, 20]) is True


def test_global_semantics():
    # Example invariant: the sum of rank values equals 100
    def sum_invariant(vals): return sum(vals) == 100
    
    assert validate_global_semantics([30, 40, 30], sum_invariant) is True
    assert validate_global_semantics([30, 40, 40], sum_invariant) is False


def test_sharded_semantics():
    global_shape = (10, 10)
    
    # Valid shards covering 10x10 exactly
    valid_shards = [
        {'offset': (0, 0), 'shape': (10, 5)},
        {'offset': (0, 5), 'shape': (10, 5)}
    ]
    assert validate_sharded_semantics(valid_shards, global_shape) is True
    
    # Invalid: missing coverage (only covers 10x8)
    invalid_shards = [
        {'offset': (0, 0), 'shape': (10, 5)},
        {'offset': (0, 5), 'shape': (10, 3)}
    ]
    assert validate_sharded_semantics(invalid_shards, global_shape) is False
