"""The head set, the loss and the ranking score must all follow one config.

NuRec's label is ``NC x DAC x GT x EP``.  A head on a term the label does not
carry learns a constant, and a term the score does not multiply cannot change a
ranking -- so all three have to be driven by the same declaration.  These tests
fail if any of them drifts back to a hard-coded EPDMS term list.
"""

import torch
import torch.nn as nn

from navsim.agents.drivesuprim.drivesuprim_config import DriveSuprimConfig
from navsim.agents.drivesuprim.drivesuprim_model import _rank_score, _sub_score_heads
from navsim.agents.drivesuprim.drivesuprim_loss_fn import _sub_score_losses

NUREC_TERMS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "gt_compliance",
    "ego_progress",
)


def _nurec_config() -> DriveSuprimConfig:
    config = DriveSuprimConfig()
    config.pdm_heads = NUREC_TERMS
    config.trajectory_pdm_weight = {
        "no_at_fault_collisions": 3.0,
        "drivable_area_compliance": 3.0,
        "gt_compliance": 3.0,
        "ego_progress": 2.0,
    }
    config.pdm_score_log = {
        "no_at_fault_collisions": 0.5,
        "drivable_area_compliance": 0.5,
        "gt_compliance": 0.5,
    }
    config.pdm_score_sum = {"ego_progress": 1.0}
    config.pdm_score_sum_scale = 6.0
    config.pdm_three_class_terms = ()
    return config


def test_only_the_configured_heads_exist():
    heads = _sub_score_heads(_nurec_config(), d_model=8, d_ffn=16, with_imi=True)

    assert set(heads) == set(NUREC_TERMS) | {"imi"}
    assert "lane_keeping" not in heads
    assert "time_to_collision_within_bound" not in heads


def test_default_config_still_builds_the_epdms_heads():
    heads = _sub_score_heads(DriveSuprimConfig(), d_model=8, d_ffn=16, with_imi=False)

    assert "lane_keeping" in heads
    assert "time_to_collision_within_bound" in heads
    assert "imi" not in heads


def test_rank_score_uses_only_configured_terms():
    config = _nurec_config()
    result = {name: torch.zeros(2, 5) for name in NUREC_TERMS}
    result["lane_keeping"] = torch.full((2, 5), 1e3)   # would dominate if read

    baseline = _rank_score(config, result, use_traffic_light=False)
    result["lane_keeping"] = torch.full((2, 5), -1e3)

    torch.testing.assert_close(_rank_score(config, result, use_traffic_light=False), baseline)


def test_rank_score_falls_with_a_multiplicative_term():
    config = _nurec_config()
    good = {name: torch.full((1, 1), 5.0) for name in NUREC_TERMS}
    voided = dict(good)
    voided["gt_compliance"] = torch.full((1, 1), -5.0)

    assert _rank_score(config, voided, False).item() < _rank_score(config, good, False).item()


def test_loss_reads_no_target_for_an_unconfigured_term():
    config = _nurec_config()
    predictions = {name: torch.zeros(2, 5, requires_grad=True) for name in NUREC_TERMS}
    # A label file still carries every column; the loss must ignore the extras.
    targets = {name: torch.ones(2, 5) for name in NUREC_TERMS}
    targets["lane_keeping"] = torch.zeros(2, 5)

    total, per_term = _sub_score_losses(predictions, targets, config, torch.float32)

    assert set(per_term) == set(NUREC_TERMS)
    total.backward()
    assert all(p.grad is not None for p in predictions.values())


def test_a_missing_target_column_costs_no_supervision():
    config = _nurec_config()
    predictions = {name: torch.zeros(2, 5, requires_grad=True) for name in NUREC_TERMS}
    targets = {name: torch.ones(2, 5) for name in NUREC_TERMS if name != "gt_compliance"}

    _, per_term = _sub_score_losses(predictions, targets, config, torch.float32)

    assert per_term["gt_compliance"].item() == 0.0
    assert per_term["ego_progress"].item() > 0.0


def _with_aggregate() -> DriveSuprimConfig:
    config = _nurec_config()
    config.pdm_aggregate_head = True
    config.pdm_aggregate_loss_weight = 1.0
    config.pdm_aggregate_rank_weight = 1.0
    return config


def test_aggregate_head_is_built_and_supervised():
    config = _with_aggregate()
    heads = _sub_score_heads(config, d_model=8, d_ffn=16, with_imi=True)
    assert "pdm_score" in heads

    predictions = {n: torch.zeros(2, 5, requires_grad=True) for n in (*NUREC_TERMS, "pdm_score")}
    targets = {n: torch.ones(2, 5) for n in (*NUREC_TERMS, "pdm_score")}
    _, per_term = _sub_score_losses(predictions, targets, config, torch.float32)

    assert per_term["pdm_score"].item() > 0.0


def test_aggregate_head_is_absent_when_off():
    heads = _sub_score_heads(_nurec_config(), d_model=8, d_ffn=16, with_imi=True)
    assert "pdm_score" not in heads


def test_aggregate_ranks_but_cannot_veto():
    """A log term at sigmoid->0 goes to -inf and overrides everything; a bonus
    bounded by its weight cannot."""
    config = _with_aggregate()
    good = {n: torch.full((1, 1), 5.0) for n in (*NUREC_TERMS, "pdm_score")}
    poor = dict(good)
    poor["pdm_score"] = torch.full((1, 1), -20.0)

    drop = _rank_score(config, good, False).item() - _rank_score(config, poor, False).item()

    assert 0.0 < drop <= config.pdm_aggregate_rank_weight + 1e-6
