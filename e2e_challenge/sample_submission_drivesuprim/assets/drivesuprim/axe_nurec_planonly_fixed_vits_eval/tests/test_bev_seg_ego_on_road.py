"""Frames whose map misses the road under the car must not train the seg head.

Measured over 1,000 frames from 250 clips, 0.4% carry a drivable target with
nothing within six metres ahead of the rear axle -- the ego is on tarmac the
lane polygons do not describe. Supervising those teaches the head that the road
the car is driving on is not road.

They are dropped by filling the target with -1 rather than by a new branch in
the loss: ``_bev_seg_loss`` already keeps only labels inside
``[0, num_classes)`` and averages over the survivors, so an all -1 frame
contributes nothing while every other frame in the batch is unaffected. Its
trajectory labels are untouched, so the frame stays in the split and stages 2
and 3 still learn from it.
"""

import numpy as np
import pytest
import torch

from navsim.agents.drivesuprim.drivesuprim_config import DriveSuprimConfig
from navsim.agents.drivesuprim.drivesuprim_features import DriveSuprimTargetBuilder
from navsim.agents.drivesuprim.drivesuprim_loss_fn import _bev_seg_loss


def _builder(**overrides):
    return DriveSuprimTargetBuilder(DriveSuprimConfig(training=False, **overrides))


def _mask(builder, cells=()):
    m = np.zeros((builder._config.bev_h, builder._config.bev_w), dtype=bool)
    for row, col in cells:
        m[row, col] = True
    return m


def _cell(builder, x, y):
    """Metric ego coordinates -> (row, col), whatever grid the config uses."""
    col, row = builder._coords_to_pixel(np.array([[[x, y]]], dtype=float)).reshape(2)
    return int(row), int(col)


def _ego_cell(builder):
    return _cell(builder, 0.0, 0.0)


def test_road_under_the_car_passes():
    builder = _builder(aux_bev_seg_require_ego_on_road=True)
    assert builder._ego_is_on_road(_mask(builder, [_cell(builder, 2.0, 0.0)]))


def test_road_only_far_away_fails():
    """The 52f5460fc9 shape: a patch at the far corner, nothing at the ego."""
    builder = _builder(aux_bev_seg_require_ego_on_road=True)
    x_max, y_max = builder._config.point_cloud_range[3], builder._config.point_cloud_range[4]
    far = [_cell(builder, x, y)
           for x in np.linspace(0.7 * x_max, 0.95 * x_max, 8)
           for y in np.linspace(0.5 * y_max, 0.9 * y_max, 8)]
    assert not builder._ego_is_on_road(_mask(builder, far))


def test_an_empty_mask_fails():
    builder = _builder(aux_bev_seg_require_ego_on_road=True)
    assert not builder._ego_is_on_road(_mask(builder))


def test_road_beyond_the_band_fails():
    """Just past the band is still a gap under the car."""
    builder = _builder(aux_bev_seg_require_ego_on_road=True, aux_bev_seg_ego_band_m=6.0)
    assert not builder._ego_is_on_road(_mask(builder, [_cell(builder, 12.0, 0.0)]))
    assert builder._ego_is_on_road(_mask(builder, [_cell(builder, 3.0, 0.0)]))


def test_road_off_to_the_side_fails():
    """A parallel carriageway ten metres away is not the road under the car."""
    builder = _builder(aux_bev_seg_require_ego_on_road=True, aux_bev_seg_ego_half_width_m=2.0)
    assert not builder._ego_is_on_road(_mask(builder, [_cell(builder, 3.0, 10.0)]))


def test_the_gate_is_off_by_default():
    """NAVSIM agents set nothing here and must keep supervising every frame."""
    builder = _builder()
    assert builder._config.aux_bev_seg_require_ego_on_road is False
    assert builder._ego_is_on_road(_mask(builder))       # even an empty mask


def test_an_all_negative_target_contributes_nothing_to_the_loss():
    """The mechanism the drop relies on."""
    config = DriveSuprimConfig(training=False)
    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    dropped = torch.full((1, 8, 8), -1.0)
    kept = torch.zeros((1, 8, 8))
    kept[0, :4] = 1.0

    both = _bev_seg_loss(logits, torch.cat([dropped, kept]), config, None)
    kept_only = _bev_seg_loss(logits[1:], kept, config, None)
    torch.testing.assert_close(both, kept_only)


def test_a_batch_of_only_dropped_frames_keeps_the_graph_alive():
    """A zero that still has a grad_fn -- DDP dies on a head that gets none."""
    config = DriveSuprimConfig(training=False)
    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    loss = _bev_seg_loss(logits, torch.full((2, 8, 8), -1.0), config, None)
    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0
