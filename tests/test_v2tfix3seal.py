"""V2T-Fix3-Seal evidence tests: provenance, weight health, corrected
GAT factorial, full tensor parity, resume reproducibility."""
from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
OUT = ROOT / "outputs/official_sgaligner_migration_fix2_v2tfix3_seal"


def load(name):
    return json.loads((OUT / name).read_text())


class TestCheckpointProvenance(unittest.TestCase):
    def test_byte_identical_redownload(self):
        p = load("checkpoint_provenance.json")
        self.assertTrue(p["comparison"]["cmp_byte_identical"])
        self.assertTrue(p["comparison"]["sha256_equal"])
        self.assertEqual(
            p["current_checkpoint"]["sha256"],
            p["fresh_download"]["sha256"],
        )
        self.assertEqual(
            p["current_checkpoint"]["recorded_sha256_file"],
            p["current_checkpoint"]["sha256"],
        )

    def test_official_folder_enumerated_single_checkpoint(self):
        p = load("checkpoint_provenance.json")
        files = p["official_source"]["files_in_official_folder"]
        self.assertEqual(len(files), 5)
        ckpts = [n for n in files.values() if n.endswith(".pth.tar")]
        self.assertEqual(
            ckpts, ["sgaligner_pct_gat_rel_attr.pth.tar"])
        self.assertFalse(
            p["official_source"][
                "other_pct_gat_rel_attr_or_best_snapshot_found"])

    def test_schema_metadata_recorded(self):
        p = load("checkpoint_provenance.json")
        self.assertEqual(p["schema"]["epoch"], 6)
        self.assertEqual(p["schema"]["iteration"], 49806)
        self.assertEqual(p["schema"]["model_num_tensors"], 92)
        self.assertTrue(p["fresh_download"]["torch_load_ok"])
        # torch.load alone must NOT be the basis of the conclusion
        self.assertTrue(p["comparison"]["cmp_byte_identical"])


class TestWeightHealth(unittest.TestCase):
    def test_four_models_audited(self):
        h = load("checkpoint_weight_health.json")
        for label in ("A_current_official", "B_redownloaded_official",
                      "C_fix1_epoch21", "D_random_init_control"):
            self.assertIn(label, h)
            self.assertGreater(h[label]["num_tensors"], 80)
            for module in ("object_encoder", "structure_encoder",
                           "structure_embedding", "meta_embedding_rel",
                           "fusion"):
                self.assertIn(module, h[label]["module_summary"])

    def test_gat_branch_degenerate_in_official_checkpoint(self):
        h = load("checkpoint_weight_health.json")
        for label in ("A_current_official", "B_redownloaded_official",
                      "C_fix1_epoch21"):
            for key, stats in h[label]["per_tensor"].items():
                if key.startswith("structure_encoder.layer_stack"):
                    self.assertLess(
                        stats["max"], 1e-30,
                        f"{label}/{key} unexpectedly healthy")

    def test_random_control_gat_healthy(self):
        h = load("checkpoint_weight_health.json")
        d = h["D_random_init_control"]["per_tensor"]
        for key, stats in d.items():
            if key.startswith("structure_encoder.layer_stack"):
                if key.endswith("lin.weight"):
                    self.assertGreater(stats["max"], 1e-3)

    def test_a_b_identical_c_differs_where_expected(self):
        diff = load("checkpoint_tensor_diff.json")
        ab = diff["A_vs_B_current_vs_redownload"]
        self.assertEqual(ab["changed_keys"], [])
        self.assertEqual(ab["max_abs_diff_overall"], 0.0)
        ac = diff["A_vs_C_official_vs_fix1_epoch21"]
        # relation head trained; PCT BN buffers drifted (pre-Fix2
        # behaviour, quantified here); GAT untouched
        changed = set(ac["changed_keys"])
        self.assertIn("meta_embedding_rel.weight", changed)
        self.assertTrue(any("running_var" in k for k in changed))
        self.assertFalse(any(
            k.startswith("structure_encoder") for k in changed))


class TestCorrectedGatFactorial(unittest.TestCase):
    def test_explicit_lt_complete_assertion(self):
        f = load("gat_factorial_corrected.json")
        self.assertTrue(f["assertion_explicit_lt_complete"])
        self.assertEqual(f["complete_graph_special_cases"], [])
        self.assertEqual(f["errors"], [])

    def test_bug_actually_fixed_edges_differ(self):
        f = load("gat_factorial_corrected.json")
        rows = f["rows"]
        for r in rows[:50]:
            if r["adjacency"] == "explicit_only_true":
                self.assertLess(
                    r["explicit_edge_count"],
                    r["complete_edge_count"])
                break

    def test_both_models_in_every_combo(self):
        f = load("gat_factorial_corrected.json")
        combos = set()
        for key in f["summary"]:
            parts = key.split("|")
            combos.add(tuple(parts))
        base = {(m, a, r) for m, a, r, _ in combos}
        for m, a, r in base:
            self.assertIn((m, a, r, "ckpt"), combos)
            self.assertIn((m, a, r, "random_init"), combos)

    def test_ckpt_collapses_random_discriminates(self):
        f = load("gat_factorial_corrected.json")
        s = f["summary"]
        ckpt_uniq = {
            v["mean_unique"] for k, v in s.items() if k.endswith("|ckpt")}
        self.assertEqual(ckpt_uniq, {1.0})  # single unique embedding
        random_explicit = [
            v["mean_unique"] for k, v in s.items()
            if "explicit_only_true" in k and k.endswith("|random_init")]
        self.assertTrue(
            all(u > 10 for u in random_explicit), random_explicit)

    def test_stagewise_audit_present(self):
        st = load("gat_stagewise_audit.json")
        self.assertGreaterEqual(len(st["rows"]), 20)
        stages = set(st["rows"][0]["per_stage"].keys())
        self.assertTrue({
            "gat_layer0", "gat_layer1", "structure_embedding",
            "normalized"} <= stages)


class TestFullTensorParity(unittest.TestCase):
    def test_deterministic_fields_exact(self):
        p = load("official_adapter_full_tensor_parity.json")
        s = p["summary"]
        for field in ("object_id_set", "root_obj_id", "rel_trans",
                      "explicit_edges", "explicit_edges_raw_order",
                      "complete_none_edges_canonical", "bow_edge_41d",
                      "bow_attr_164d", "graph_counts"):
            self.assertEqual(
                s[field]["equal"], s[field]["n"], field)
            self.assertEqual(s[field]["max_abs_diff_overall"], 0.0,
                             field)

    def test_points_mismatch_honestly_reported(self):
        p = load("official_adapter_full_tensor_parity.json")
        s = p["summary"]
        # RNG provenance makes byte equality impossible; the evidence
        # is the documented reason + geometry agreement
        self.assertEqual(s["points_512_exact"]["equal"], 0)
        for row in p["rows"]:
            g = row["points_512_geometry_agreement"]
            self.assertLess(g["per_object_mean_nn_dist_mean"], 0.05)
            self.assertIn("RNG", row["points_512_exact"]["note"]
                          + row["points_512_rowsort_hash"]["note"])
            # every field carries BOTH hashes
            self.assertTrue(row["points_512_exact"]["official_hash"])
            self.assertTrue(row["points_512_exact"]["adapter_hash"])
            self.assertTrue(row["rel_trans"]["official_hash"])
            self.assertTrue(row["bow_edge_41d"]["official_hash"])
            self.assertTrue(row["bow_attr_164d"]["adapter_hash"])

    def test_every_field_reports_first_mismatch_or_equal(self):
        p = load("official_adapter_full_tensor_parity.json")
        fields = ("object_id_set", "root_obj_id", "rel_trans",
                  "explicit_edges", "complete_none_edges_canonical",
                  "bow_edge_41d", "bow_attr_164d", "graph_counts",
                  "points_512_exact")
        for row in p["rows"]:
            if "error" in row:
                continue
            for field in fields:
                self.assertIn("equal", row[field])
                self.assertIn(
                    "first_mismatch", row[field])
                self.assertIn("max_abs_diff", row[field])


class TestResumeProtocol(unittest.TestCase):
    def test_script_in_repo_not_tmp(self):
        self.assertTrue((ROOT / "scripts/v2f3_resume_protocol.py").exists())

    def test_cpu_synthetic_exact(self):
        r = load("resume/resume_equivalence_cpu.json")
        self.assertEqual(r["model"]["changed"], 0)
        self.assertEqual(r["model"]["max_diff"], 0.0)
        self.assertEqual(r["optimizer"]["changed"], 0)
        self.assertTrue(r["scheduler_identical"])
        self.assertTrue(r["history_identical"])
        self.assertTrue(r["lr_identical"])
        self.assertTrue(r["exact"])
        self.assertTrue(r["numpy_rng_identical"])

    def test_gpu_real_subset_exact(self):
        r = load("resume/resume_equivalence_gpu.json")
        self.assertEqual(r["model"]["changed"], 0)
        self.assertEqual(r["model"]["max_diff"], 0.0)
        self.assertEqual(r["optimizer"]["changed"], 0)
        self.assertTrue(r["exact"])

    def test_failclosed_refusals(self):
        r = load("resume/resume_failclosed.json")
        self.assertTrue(r["total_epochs_mismatch"]["refused"])
        self.assertTrue(r["dataset_fingerprint_mismatch"]["refused"])

    def test_commands_reference_only_repo_scripts(self):
        sh = (OUT / "commands.sh").read_text()
        self.assertNotIn("/tmp/resume_protocol.py", sh)
        for line in sh.splitlines():
            if "python" in line and "scripts/" in line:
                self.assertNotIn("/tmp/", line)


class TestFinalDecision(unittest.TestCase):
    def test_case_a_wording_no_architecture_blame(self):
        d = (OUT / "final_decision.md").read_text()
        flat = " ".join(d.split())  # markdown line-wrap agnostic
        self.assertIn("情况A", flat)
        self.assertIn("数值退化", flat)
        for banned in ("架构天然坍缩", "MultiGAT架构天然"):
            self.assertNotIn(banned, flat)
        self.assertIn("research branch", flat)


if __name__ == "__main__":
    unittest.main()
