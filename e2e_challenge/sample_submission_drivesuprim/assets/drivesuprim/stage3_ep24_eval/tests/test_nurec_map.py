import pickle

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from navsim.planning.data.nurec_attach_maps import attach_map, write_index
from navsim.planning.data.nurec_map import NuRecMap, load_nurec_map


def _bundle():
    lane_a = {
        "left": [[0, 1, 0], [10, 1, 0]],
        "right": [[0, -1, 0], [10, -1, 0]],
        "center": [[0, 0, 0], [10, 0, 0]],
        "next": ["lane-b"],
        "previous": [],
        "left_lane": [],
        "right_lane": [],
        "roadblock_id": "nurec-rb:a",
        "speed_limit_mps": 10.0,
    }
    lane_b = {
        "left": [[10, 1, 0], [20, 1, 0]],
        "right": [[10, -1, 0], [20, -1, 0]],
        "center": [[10, 0, 0], [20, 0, 0]],
        "next": [],
        "previous": ["lane-a"],
        "left_lane": [],
        "right_lane": [],
        "roadblock_id": "nurec-rb:b",
        "speed_limit_mps": 10.0,
    }
    return {
        "format_version": 1,
        "clip_id": "clip",
        "map_name": "nurec:clip",
        "lanes": {"lane-a": lane_a, "lane-b": lane_b},
        "roadblocks": {
            "nurec-rb:a": {
                "lane_ids": ["lane-a"], "incoming": [], "outgoing": ["nurec-rb:b"], "is_connector": False
            },
            "nurec-rb:b": {
                "lane_ids": ["lane-b"], "incoming": ["nurec-rb:a"], "outgoing": [], "is_connector": False
            },
        },
        "polygons": {"intersection_area": [], "crosswalk": [], "drivable_space": []},
        "matched_ego": [
            {"timestamp_us": 0, "lane_id": "lane-a", "roadblock_id": "nurec-rb:a"},
            {"timestamp_us": 500_000, "lane_id": "lane-b", "roadblock_id": "nurec-rb:b"},
        ],
        "route_roadblock_ids": ["nurec-rb:a", "nurec-rb:b"],
    }


def test_nurec_map_exposes_lane_and_roadblock_graph(tmp_path, monkeypatch):
    bundle = _bundle()
    with (tmp_path / "clip.pkl").open("wb") as stream:
        pickle.dump(bundle, stream)
    monkeypatch.setenv("NUREC_MAP_ROOT", str(tmp_path))

    map_api = load_nurec_map("nurec:clip")
    first = map_api.get_map_object("nurec-rb:a", SemanticMapLayer.ROADBLOCK)
    assert [edge.id for edge in first.outgoing_edges] == ["nurec-rb:b"]
    assert first.interior_edges[0].baseline_path.length == 10.0
    nearby = map_api.get_proximal_map_objects(Point2D(5, 0), 1, [SemanticMapLayer.LANE])
    assert [lane.id for lane in nearby[SemanticMapLayer.LANE]] == ["lane-a"]


def test_nurec_map_exposes_drivable_union_and_preserves_holes():
    bundle = _bundle()
    bundle["polygons"]["drivable_space"] = [{
        "id": "nurec-drivable-0",
        "points": [[-1, -2, 0], [21, -2, 0], [21, 2, 0], [-1, 2, 0], [-1, -2, 0]],
        "holes": [[[9, -.5, 0], [11, -.5, 0], [11, .5, 0], [9, .5, 0], [9, -.5, 0]]],
    }]
    map_api = NuRecMap(bundle)
    assert map_api.is_in_layer(Point2D(5, 0), SemanticMapLayer.DRIVABLE_AREA)
    assert not map_api.is_in_layer(Point2D(10, 0), SemanticMapLayer.DRIVABLE_AREA)


def test_attach_map_writes_future_route_suffix():
    frames = [{"timestamp": 5_000}, {"timestamp": 490_000}]
    stats = attach_map(frames, _bundle())
    assert stats["frames_with_route"] == 2
    assert frames[0]["map_location"] == "nurec:clip"
    assert frames[0]["roadblock_ids"] == ["nurec-rb:a", "nurec-rb:b"]
    assert frames[1]["roadblock_ids"] == ["nurec-rb:b"]


def test_nurec_index_keeps_all_training_logs_and_samples_validation(tmp_path):
    names = [f"log-{index:02d}" for index in range(20)]
    summary = {"log_names": names, "frames": 123}
    index_path = tmp_path / "index.json"
    write_index(
        summary,
        tmp_path / "maps",
        tmp_path / "sensors",
        index_path,
        val_fraction=0.1,
        split_seed=7,
    )

    import json

    index = json.loads(index_path.read_text())
    assert index["logs"]["train"] == names
    assert len(index["logs"]["val"]) == 2
    assert set(index["logs"]["val"]).issubset(index["logs"]["train"])
    assert index["validation_sampling"]["overlaps_train"] is True
