from types import SimpleNamespace

import numpy as np

from navsim.evaluate.pdm_score import aggregate_epdms_labels


def _config():
    return SimpleNamespace(
        progress_weight=5.0,
        ttc_weight=5.0,
        lane_keeping_weight=2.0,
        history_comfort_weight=2.0,
    )


def test_score_is_progress_gated_by_the_multiplicative_terms():
    """score = NC x DAC x GT x EP."""
    gt = {
        "no_at_fault_collisions": np.asarray([1.0, 1.0, 0.0], dtype=np.float16),
        "drivable_area_compliance": np.ones(3, dtype=np.float16),
        "gt_compliance": np.ones(3, dtype=np.float16),
        "ego_progress": np.asarray([1.0, 0.5, 1.0], dtype=np.float16),
        "time_to_collision_within_bound": np.zeros(3, dtype=np.float16),
    }

    score = aggregate_epdms_labels(gt, _config())

    # progress carries the score outright; TTC = 0 no longer touches it
    np.testing.assert_allclose(score[0], 1.0)
    np.testing.assert_allclose(score[1], 0.5, rtol=1e-3)
    # a collision zeroes the product whatever progress says
    np.testing.assert_allclose(score[2], 0.0)


def test_dropped_metrics_do_not_enter_the_score():
    """All still exported; none sets a label (see the docstring for why)."""
    base = {
        "no_at_fault_collisions": np.ones(1, dtype=np.float16),
        "drivable_area_compliance": np.ones(1, dtype=np.float16),
        "gt_compliance": np.ones(1, dtype=np.float16),
        "ego_progress": np.ones(1, dtype=np.float16),
        "time_to_collision_within_bound": np.ones(1, dtype=np.float16),
    }
    spoiled = dict(base)
    spoiled["time_to_collision_within_bound"] = np.zeros(1, dtype=np.float16)
    spoiled["lane_keeping"] = np.zeros(1, dtype=np.float16)
    spoiled["driving_direction_compliance"] = np.zeros(1, dtype=np.float16)
    spoiled["traffic_light_compliance"] = np.zeros(1, dtype=np.float16)
    spoiled["comfort"] = np.zeros(1, dtype=np.float16)
    spoiled["history_comfort"] = np.zeros(1, dtype=np.float16)

    config = _config()
    np.testing.assert_allclose(
        aggregate_epdms_labels(base, config), aggregate_epdms_labels(spoiled, config)
    )


def test_drift_past_four_metres_zeroes_the_score():
    gt = {
        "no_at_fault_collisions": np.ones(2, dtype=np.float16),
        "drivable_area_compliance": np.ones(2, dtype=np.float16),
        "gt_compliance": np.asarray([1.0, 0.0], dtype=np.float16),
        "ego_progress": np.ones(2, dtype=np.float16),
        "time_to_collision_within_bound": np.ones(2, dtype=np.float16),
    }

    score = aggregate_epdms_labels(gt, _config())

    np.testing.assert_allclose(score[0], 1.0)
    np.testing.assert_allclose(score[1], 0.0)


def test_labels_without_the_newer_terms_stay_readable():
    """Older label files carry no gt_compliance."""
    gt = {
        "no_at_fault_collisions": np.ones(1, dtype=np.float16),
        "drivable_area_compliance": np.ones(1, dtype=np.float16),
        "ego_progress": np.ones(1, dtype=np.float16),
        "time_to_collision_within_bound": np.ones(1, dtype=np.float16),
    }

    np.testing.assert_allclose(aggregate_epdms_labels(gt, _config()), 1.0)
