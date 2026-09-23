"""The drivable target's layers must survive the trip through YAML.

Hydra hands a YAML list over as strings, and a string matches no map layer:
``get_proximal_map_objects`` returns nothing for it, the mask comes out entirely
zero, and training proceeds happily against a blank map. That happened once and
was only caught because a separate measurement printed the coverage. The
resolution is pinned here, and an unknown name has to raise rather than vanish.

The layer choice itself is the other half. LANE + INTERSECTION -- the value
inherited from TransfuserConfig -- covers a roundabout with a single polygon,
centre island included; the NuRec agents use LANE + LANE_CONNECTOR, which is
also what ``pdm_scorer._road_edge_compliance`` reads when deciding DAC.
"""

import pytest
import yaml
from pathlib import Path

from nuplan.common.maps.abstract_map import SemanticMapLayer

from navsim.agents.drivesuprim.drivesuprim_config import DriveSuprimConfig
from navsim.agents.drivesuprim.drivesuprim_features import DriveSuprimTargetBuilder

AGENT_DIR = Path(__file__).resolve().parents[1] / "navsim/planning/script/config/common/agent"
NUREC_AGENTS = [
    "drivesuprim_agent_bevformer_vov_v2_vits_planonly_scratch_nurec",
    "drivesuprim_agent_bevformer_vov_v2_vits_stage1_nurec",
    "drivesuprim_agent_bevformer_vov_v2_vits_stage2_nurec",
    "drivesuprim_agent_bevformer_vov_v2_vits_stage3_nurec",
]


def _builder(**overrides):
    # the drivable mask is a TARGET, so the helper lives on the target builder
    return DriveSuprimTargetBuilder(DriveSuprimConfig(training=False, **overrides))


def test_layer_names_from_yaml_resolve_to_enum_members():
    builder = _builder(aux_bev_drivable_layers=["LANE", "LANE_CONNECTOR"])
    assert builder._drivable_layers() == [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]


def test_enum_members_pass_through_unchanged():
    layers = [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]
    assert _builder(aux_bev_drivable_layers=layers)._drivable_layers() == layers


def test_an_unknown_layer_name_raises():
    """Silently matching nothing is what produced an all-zero target."""
    builder = _builder(aux_bev_drivable_layers=["LANE", "ROAD_SURFACE"])
    with pytest.raises(ValueError, match="ROAD_SURFACE"):
        builder._drivable_layers()


def test_the_default_is_the_inherited_transfuser_choice():
    """NAVSIM agents set nothing here and must keep their behaviour."""
    builder = _builder()
    assert builder._config.aux_bev_drivable_layers is None
    assert builder._drivable_layers() == list(builder._config.bev_semantic_classes[1][1])
    assert SemanticMapLayer.INTERSECTION in builder._drivable_layers()


@pytest.mark.parametrize("name", NUREC_AGENTS)
def test_every_nurec_agent_asks_for_the_lane_layers(name):
    with (AGENT_DIR / f"{name}.yaml").open() as fh:
        config = yaml.safe_load(fh)["config"]
    assert config["aux_bev_drivable_layers"] == ["LANE", "LANE_CONNECTOR"]
    # INTERSECTION is what paints roundabout centre islands as drivable
    assert "INTERSECTION" not in config["aux_bev_drivable_layers"]


@pytest.mark.parametrize("name", NUREC_AGENTS)
def test_every_nurec_agent_weighs_drivable_as_the_minority_class(name):
    """Measured over the whole training split, 145.7M cells: background 64.469%, drivable 35.531%."""
    with (AGENT_DIR / f"{name}.yaml").open() as fh:
        weights = yaml.safe_load(fh)["config"]["aux_bev_seg_class_weights"]
    background, drivable = weights
    assert drivable > background, "drivable is the minority class on NuRec"
    assert weights == pytest.approx([0.7106, 1.2894])
    assert background + drivable == pytest.approx(2.0, abs=1e-3), "weights average to 1"


@pytest.mark.parametrize("name", NUREC_AGENTS)
def test_the_nurec_agents_agree_with_each_other(name):
    def block(agent):
        with (AGENT_DIR / f"{agent}.yaml").open() as fh:
            c = yaml.safe_load(fh)["config"]
        return c["aux_bev_drivable_layers"], c["aux_bev_seg_class_weights"]

    assert block(name) == block(NUREC_AGENTS[0])
