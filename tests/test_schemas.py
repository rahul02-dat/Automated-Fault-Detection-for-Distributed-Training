import os
import json
import pytest
from dataclasses import dataclass
from typing import Dict, Any

from src.experiments.result_schema import ExperimentResult, ExperimentPhase, ExperimentStatus, ExperimentOutcome
from src.experiments.experiment_manifest import ExperimentManifest
from research.state_model import OwnershipScope
from research.semantic_contract import StateContract

def test_experiment_result_round_trip(tmp_path):
    res = ExperimentResult(
        experiment_id="test_exp",
        run_id="run_1",
        workload="test_workload",
        phase=ExperimentPhase.FAULT_INJECTION.value,
        status=ExperimentStatus.PASS.value,
        detected=True,
        expected_contracts=["model"],
        observed_contracts=["model"],
        root_cause="model",
        downstream_effects=["optimizer"],
        mutation_applied=True,
        validation_mode="digest_first",
        resumed_metric=0.123
    )
    
    path = os.path.join(tmp_path, "res.json")
    res.save(path)
    
    loaded_res = ExperimentResult.load(path)
    
    assert loaded_res.experiment_id == res.experiment_id
    assert loaded_res.phase == res.phase
    assert loaded_res.status == res.status
    assert loaded_res.expected_contracts == res.expected_contracts
    assert loaded_res.downstream_effects == res.downstream_effects
    assert loaded_res.mutation_applied == res.mutation_applied
    assert loaded_res.validation_mode == res.validation_mode
    assert loaded_res.resumed_metric == res.resumed_metric
    
    # Check serialization semantic equality
    with open(path, "r") as f:
        data = json.load(f)
    assert data["experiment_id"] == "test_exp"
    assert data["validation_mode"] == "digest_first"


def test_experiment_manifest_round_trip(tmp_path):
    manifest = ExperimentManifest(
        experiment_id="test_exp",
        run_id="run_1",
        comparator="ALLCLOSE",
        rtol=1e-5,
        atol=1e-8,
        world_size=4
    )
    
    path = os.path.join(tmp_path, "manifest.json")
    manifest.save(path)
    
    loaded_manifest = ExperimentManifest.load(path)
    
    assert loaded_manifest.experiment_id == manifest.experiment_id
    assert loaded_manifest.comparator == manifest.comparator
    assert loaded_manifest.rtol == manifest.rtol
    assert loaded_manifest.atol == manifest.atol
    assert loaded_manifest.world_size == manifest.world_size


def test_experiment_result_missing_required_fields():
    with pytest.raises(TypeError):
        # Missing experiment_id
        res = ExperimentResult()


def test_experiment_manifest_missing_required_fields():
    with pytest.raises(TypeError):
        manifest = ExperimentManifest()


def test_state_contract_invalid_scope():
    with pytest.raises(ValueError):
        StateContract(name="model", scope="INVALID_SCOPE", comparator="EXACT")
