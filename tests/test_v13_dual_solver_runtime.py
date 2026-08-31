import tempfile
from pathlib import Path
import unittest

import numpy as np

from safety.v13_dual_solver_runtime import (
    REPEATS, PointDSCSolver, RuntimeContractError, SolverFailure,
    apply_transform, complete_linkage_q4,
    fixed_permutation, load_frozen_correspondences, pygcransac_row_to_column,
    solve_pygcransac, summarize_workers, transform_distance, worker_payload,
)


def known_transform():
    angle = np.deg2rad(23.0)
    c, s = np.cos(angle), np.sin(angle)
    out = np.eye(4)
    out[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    out[:3, 3] = [0.31, -0.17, 0.08]
    return out


class Tests(unittest.TestCase):
    def test_cache_is_exact_stable_top1000(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corr.npz"
            src = np.arange(3600, dtype=np.float64).reshape(1200, 3) / 1000
            ref = src + 1
            scores = np.ones(1200, np.float64)
            scores[1001] = 2
            np.savez(path, src_corr=src, ref_corr=ref, scores=scores)
            data = load_frozen_correspondences(path)
            self.assertEqual(len(data.src), 1000)
            self.assertEqual(int(data.selected_original_indices[0]), 1001)
            self.assertTrue(np.array_equal(data.selected_original_indices[1:6], range(5)))
            bad = Path(tmp) / "bad.npz"
            np.savez(bad, src_corr=src, ref_corr=ref, scores=scores, gt_transform=np.eye(4))
            with self.assertRaisesRegex(RuntimeContractError, "exactly"):
                load_frozen_correspondences(bad)

    def test_fixed_permutations_are_distinct_and_reproducible(self):
        values = [fixed_permutation("a" * 64, "forward", i, 100)[0] for i in range(5)]
        repeat = [fixed_permutation("a" * 64, "forward", i, 100)[0] for i in range(5)]
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(values, repeat)))
        self.assertEqual(len({x.tobytes() for x in values}), 5)

    def test_pygcransac_raw_row_pose_must_transpose(self):
        transform = known_transform()
        raw_row_pose = transform.T
        converted = pygcransac_row_to_column(raw_row_pose)
        self.assertTrue(np.allclose(converted, transform))
        point = np.array([[0.3, -0.2, 0.7]])
        expected = point @ raw_row_pose[:3, :3] + raw_row_pose[3, :3]
        self.assertTrue(np.allclose(apply_transform(point, converted), expected))
        with self.assertRaisesRegex(Exception, "homogeneous"):
            pygcransac_row_to_column(transform)

    def test_real_pygcransac_nonzero_se3(self):
        rng = np.random.default_rng(7)
        src = rng.uniform(-1, 1, (120, 3))
        expected = known_transform()
        ref = apply_transform(src, expected)
        ref[90:] = rng.uniform(-2, 2, (30, 3))
        got, diagnostics = solve_pygcransac(src, ref)
        dr, dt = transform_distance(got, expected)
        self.assertLess(dr, 0.2)
        self.assertLess(dt, 0.005)
        self.assertGreaterEqual(diagnostics["inlier_count"], 80)

    def test_real_sealed_pointdsc_nonzero_se3(self):
        root = Path("/home/aidenwu/Documents/SceneGraphFusion_RGBDPointDSC/upstream/PointDSC")
        checkpoint = root / "snapshot/PointDSC_3DMatch_release/models/model_best.pkl"
        if not checkpoint.is_file():
            self.skipTest("sealed PointDSC checkout unavailable")
        rng = np.random.default_rng(8)
        src = rng.uniform(-1, 1, (120, 3))
        expected = known_transform()
        ref = apply_transform(src, expected)
        ref[90:] = rng.uniform(-2, 2, (30, 3))
        got, diagnostics = PointDSCSolver(root, checkpoint, "cpu").solve(src, ref)
        dr, dt = transform_distance(got, expected)
        self.assertLess(dr, 0.2)
        self.assertLess(dt, 0.005)
        self.assertGreaterEqual(diagnostics["inlier_count"], 80)

    def row(self, solver, direction, repeat, transform):
        return {"solver": solver, "direction": direction, "repeat": repeat,
                "status": "ok", "transform": np.asarray(transform).tolist()}

    def test_q4_and_cross_direction_cross_solver(self):
        transform = known_transform()
        rows = []
        for solver in ("pointdsc", "pygcransac"):
            for direction in ("forward", "reverse"):
                value = transform if direction == "forward" else np.linalg.inv(transform)
                rows.extend(self.row(solver, direction, repeat, value) for repeat in range(REPEATS))
        gate = complete_linkage_q4(rows[:5])
        self.assertTrue(gate["usable"])
        result = summarize_workers(rows)
        self.assertTrue(result["safe"])
        self.assertFalse(result["registration_decision_authorized"])
        veto = summarize_workers(rows, known_bad=True)
        self.assertFalse(veto["safe"])
        self.assertEqual(veto["reason"], "known_bad_veto")
        rows[0]["status"] = "failed"; rows[0]["transform"] = None
        rows[1]["status"] = "failed"; rows[1]["transform"] = None
        self.assertFalse(summarize_workers(rows)["safe"])

    def test_failed_worker_has_no_identity_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corr.npz"
            points = np.arange(120, dtype=np.float64).reshape(40, 3) / 100
            np.savez(path, src_corr=points, ref_corr=points,
                     scores=np.ones(40, np.float64))
            cache = load_frozen_correspondences(path)
            def fail(_src, _ref):
                raise SolverFailure("synthetic_failure")
            row = worker_payload(cache, "pygcransac", "forward", 0, fail, "a" * 64,
                                 {"version": "0.1.1", "implementation_sha256": "b" * 64})
            self.assertEqual(row["status"], "failed")
            self.assertIsNone(row["transform"])
            self.assertFalse(row["fallback_used"])
            self.assertEqual(row["dependency"]["version"], "0.1.1")

    def test_reverse_worker_consumes_reverse_cache_without_swapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            forward_path = Path(tmp) / "forward.npz"
            reverse_path = Path(tmp) / "reverse.npz"
            points = np.arange(120, dtype=np.float64).reshape(40, 3) / 100
            np.savez(forward_path, src_corr=points, ref_corr=points + 1,
                     scores=np.ones(40, np.float64))
            np.savez(reverse_path, src_corr=points + 7, ref_corr=points + 11,
                     scores=np.ones(40, np.float64))
            reverse = load_frozen_correspondences(reverse_path)
            observed = {}
            def capture(src, ref):
                observed["src"] = src.copy(); observed["ref"] = ref.copy()
                return np.eye(4), {"inlier_count": len(src), "inlier_ratio": 1.0}
            row = worker_payload(reverse, "pygcransac", "reverse", 0, capture,
                                 "a" * 64, {"version": "0.1.1"})
            self.assertEqual(row["status"], "ok")
            self.assertTrue(np.array_equal(observed["src"], reverse.src))
            self.assertTrue(np.array_equal(observed["ref"], reverse.ref))
            self.assertFalse(np.array_equal(observed["src"], reverse.ref))


if __name__ == "__main__":
    unittest.main()
