# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Protect the unchanged valid scores when replacing infrastructure failures."""

import copy
import unittest

from retry_disjoint512_infrastructure import validate_replacement


class ReplacementTest(unittest.TestCase):
    def setUp(self):
        self.before = {
            "rollouts": [
                {
                    "clipgt_id": str(i),
                    "rollout_id": f"old-{i}",
                    "score": 0.5,
                    "score_metrics": {"progress_score": 0.5},
                    "failure_reason": None,
                }
                for i in range(441)
            ]
        }
        self.failed = {"439", "440"}
        for row in self.before["rollouts"][-2:]:
            row.update(score=0.0, failure_reason="AioRpcError: UNAVAILABLE")
        self.after = copy.deepcopy(self.before)
        for row in self.after["rollouts"][-2:]:
            # A real model failure is retained even when its score stays zero.
            row.update(
                rollout_id="retry-" + row["clipgt_id"],
                score=0.0,
                failure_reason="collision_at_fault",
            )

    def test_completed_retry_may_have_physical_failure_and_zero_score(self):
        validate_replacement(self.before, self.after, self.failed)

    def test_changing_any_valid_score_is_rejected(self):
        self.after["rollouts"][0]["score"] = 0.6
        with self.assertRaises(AssertionError):
            validate_replacement(self.before, self.after, self.failed)

    def test_dropping_failed_clip_is_rejected(self):
        self.after["rollouts"].pop()
        with self.assertRaises(AssertionError):
            validate_replacement(self.before, self.after, self.failed)

    def test_remaining_infrastructure_failure_is_rejected(self):
        self.after["rollouts"][-1]["failure_reason"] = "AioRpcError: UNAVAILABLE"
        with self.assertRaises(AssertionError):
            validate_replacement(self.before, self.after, self.failed)

    def test_duplicate_rollout_is_rejected(self):
        self.after["rollouts"].append(copy.deepcopy(self.after["rollouts"][0]))
        with self.assertRaises(AssertionError):
            validate_replacement(self.before, self.after, self.failed)


if __name__ == "__main__":
    unittest.main()
