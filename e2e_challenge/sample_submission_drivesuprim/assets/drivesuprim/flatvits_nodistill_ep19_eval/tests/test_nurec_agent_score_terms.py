"""Every NuRec-score agent must describe the same score.

Four agent configs carry the score terms: the planning-only one the first run
used, and stage 1 / 2 / 3 of the curriculum. The block is repeated in each
rather than composed, because Hydra's defaults list is order-sensitive and a
partial config in the agent group can be selected by mistake. Repetition is the
cheaper hazard, but only if drift is caught mechanically -- which is this file.

Stage 1 matters as much as the rest even though its ``planning_loss_weight`` is
zero: stage 2 loads its weights, ``load_state_dict`` runs with ``strict=False``,
and a head set that differs between the two would be dropped in silence.
"""

from pathlib import Path

import pytest
import yaml

AGENT_DIR = Path(__file__).resolve().parents[1] / "navsim/planning/script/config/common/agent"

REFERENCE = "drivesuprim_agent_bevformer_vov_v2_vits_planonly_scratch_nurec"
STAGES = [
    "drivesuprim_agent_bevformer_vov_v2_vits_stage1_nurec",
    "drivesuprim_agent_bevformer_vov_v2_vits_stage2_nurec",
    "drivesuprim_agent_bevformer_vov_v2_vits_stage3_nurec",
]

# Everything that defines the score, and nothing that defines the stage.
SCORE_KEYS = [
    "pdm_heads",
    "trajectory_pdm_weight",
    "pdm_imi_rank_weight",
    "pdm_score_log",
    "pdm_score_sum",
    "pdm_score_sum_scale",
    "pdm_three_class_terms",
    "pdm_aggregate_head",
    "pdm_aggregate_loss_weight",
    "pdm_aggregate_rank_weight",
    "use_traffic_light_compliance",
]

DROPPED_HEADS = [
    "time_to_collision_within_bound",
    "lane_keeping",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "history_comfort",
]


def _config(name):
    with (AGENT_DIR / f"{name}.yaml").open() as fh:
        return yaml.safe_load(fh)["config"]


def _score_terms(name):
    config = _config(name)
    missing = [k for k in SCORE_KEYS if k not in config]
    assert not missing, f"{name} is missing {missing}"
    return {k: config[k] for k in SCORE_KEYS}


@pytest.mark.parametrize("name", STAGES)
def test_each_stage_matches_the_reference_agent(name):
    assert _score_terms(name) == _score_terms(REFERENCE)


def test_the_reference_scores_exactly_the_four_terms():
    terms = _score_terms(REFERENCE)
    assert terms["pdm_heads"] == [
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "gt_compliance",
        "ego_progress",
    ]
    assert terms["use_traffic_light_compliance"] is False
    assert terms["pdm_aggregate_head"] is True


@pytest.mark.parametrize("name", [REFERENCE] + STAGES)
def test_no_dropped_head_survives_anywhere(name):
    """A leftover name would rebuild a head the score cannot supervise."""
    terms = _score_terms(name)
    for dropped in DROPPED_HEADS:
        assert dropped not in terms["pdm_heads"]
        assert dropped not in terms["trajectory_pdm_weight"]
        assert dropped not in terms["pdm_score_log"]
        assert dropped not in terms["pdm_score_sum"]


@pytest.mark.parametrize("name", [REFERENCE] + STAGES)
def test_every_weighted_term_has_a_head(name):
    """A weight on a name with no head is a silent no-op, not an error."""
    terms = _score_terms(name)
    heads = set(terms["pdm_heads"])
    for field in ("trajectory_pdm_weight", "pdm_score_log", "pdm_score_sum"):
        stray = set(terms[field]) - heads
        assert not stray, f"{name}.{field} weights {sorted(stray)}, which have no head"


@pytest.mark.parametrize("name", STAGES)
def test_each_stage_writes_to_its_own_checkpoint_directory(name):
    """Sharing a directory across stages lets one stage overwrite another."""
    stage_dir = _config(name)["ckpt_path"]
    assert stage_dir.endswith("_nurec_ckpt")
    assert stage_dir != _config(REFERENCE)["ckpt_path"]
    others = [_config(o)["ckpt_path"] for o in STAGES if o != name]
    assert stage_dir not in others


@pytest.mark.parametrize("name", STAGES)
def test_each_stage_inherits_its_epdms_counterpart(name):
    """The NuRec file changes the score and nothing else about the stage."""
    with (AGENT_DIR / f"{name}.yaml").open() as fh:
        defaults = yaml.safe_load(fh)["defaults"]
    assert defaults[0] == name.removesuffix("_nurec")
    assert defaults[-1] == "_self_"
