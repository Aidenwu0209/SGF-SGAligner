import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from safety.v13_dual_solver_runtime import load_frozen_correspondences
from safety.v13_strict_pair_gate import StrictPairGateError, strict_pair_gate


PAIR="n0_to_r0";KNOWN="bad_to_ref"
SPEC={"normal_pair_ids":[PAIR,"n1_to_r1","n2_to_r2"],"known_bad_pair_id":KNOWN}


def features():
    return {"overlap_10cm":.8,"median_residual_m":.01,
            "symmetric_trimmed_chamfer_m":.01,"icp_converged":True,
            "icp_update_translation_m":0.,"icp_update_rotation_deg":0.,
            "icp_fitness":.9,"ransac_inliers":40,"spatial_extent_m":2.,
            "bidirectional_available":True,"bidirectional_rotation_deg":0.,
            "bidirectional_translation_m":0.}


class Tests(unittest.TestCase):
    def prepare(self,root,pair_id=PAIR):
        points=np.arange(180,dtype=np.float32).reshape(60,3)/100
        labels=np.repeat(np.arange(6),10)
        manifest={"schema":"v13-color-preserving-pair-v2","pair_id":pair_id,
                  "payload_sha256":"a"*64}
        prepared=root/"prepared.npz"
        np.savez(prepared,manifest_json=np.asarray(json.dumps(manifest)),
                 sgf_selected_union_source_xyz=points,
                 sgf_selected_union_source_labels=labels,
                 sgf_selected_union_reference_xyz=points,
                 sgf_selected_union_reference_labels=labels)
        caches=[]
        for direction,offset in (("forward",0.),("reverse",.001)):
            path=root/f"{direction}.npz"
            np.savez(path,src_corr=points,ref_corr=points,
                     scores=np.ones(len(points),np.float32)+offset)
            caches.append(path)
        loaded=[load_frozen_correspondences(path) for path in caches]
        summary={"schema":"v13-dual-solver-summary-v1","safe":True,
                 "cache_sha256":{"forward":loaded[0].cache_sha256,
                                  "reverse":loaded[1].cache_sha256},"gates":{}}
        for solver in ("pointdsc","pygcransac"):
            for direction in ("forward","reverse"):
                summary["gates"][f"{solver}/{direction}"]={
                    "usable":True,"medoid_transform":np.eye(4).tolist()}
        return prepared,caches,summary

    def icp(self,source,reference,initial,*,seed):
        return {"transform":np.asarray(initial),"converged":True,"fitness":1.,
                "rmse_m":0.,"update_rotation_deg":0.,"update_translation_m":0.,
                "trace":[{"fixed_correspondence_rmse_before_m":0.,
                          "fixed_correspondence_rmse_after_m":0.,
                          "update_rotation_deg":0.,"update_translation_m":0.}]}

    def rule(self,*args,**kwargs):
        return features(),{"rejection_reasons":[],"usable_for_reconstruction":True}

    def test_strict_gate_pass_and_knownbad_auto_veto(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);prepared,caches,summary=self.prepare(root)
            result=strict_pair_gate(pair_id=PAIR,arm="sgf_selected_union",
                prepared_path=prepared,forward_cache_path=caches[0],reverse_cache_path=caches[1],
                dual_summary=summary,preregistration=SPEC,icp_fn=self.icp,
                rule_features_fn=self.rule,test_injection=True)
            self.assertFalse(result["safe"])
            self.assertEqual(result["schema"],"v13-strict-pair-gate-test-only-v1")
            self.assertEqual(result["gate_authority"],"TEST_ONLY")
            self.assertTrue(result["strict_geometry_safe_before_veto"])
            self.assertEqual(len(result["medoid_safety"]),4)
            self.assertEqual(len(result["final_consistency"]),4)
            self.assertTrue(all(row["successful_node_pairs"]==0 and
                                row["failed_node_pairs"]==0 and
                                row["node_pair_success_ratio"]==0.0 and
                                row["rule_c_claimed"] is False
                                for row in result["medoid_safety"].values()))
            self.assertTrue(all(row["segment_centre_count"]>0 and
                                len(row["segment_centres_sha256"])==64
                                for row in result["medoid_safety"].values()))
            self.assertTrue(all(row["rule_b_features"] == features() and
                                row["recorded_rule_b_decision"] == {
                                    "rejection_reasons": [],
                                    "usable_for_reconstruction": True} and
                                len(row["icp"]["trace"]) == 1
                                for row in result["medoid_safety"].values()))
            self.assertTrue(all(row["surface_source_point_count"] == 60 and
                                row["surface_reference_point_count"] == 60 and
                                len(row["surface_source_sha256"]) == 64 and
                                len(row["surface_reference_sha256"]) == 64
                                for row in result["medoid_safety"].values()))
            json.dumps(result, allow_nan=False)
            prepared,caches,summary=self.prepare(root,KNOWN)
            veto=strict_pair_gate(pair_id=KNOWN,arm="sgf_selected_union",
                prepared_path=prepared,forward_cache_path=caches[0],reverse_cache_path=caches[1],
                dual_summary=summary,preregistration=SPEC,icp_fn=self.icp,
                rule_features_fn=self.rule,test_injection=True)
            self.assertFalse(veto["safe"]);self.assertTrue(veto["known_bad_veto"])
            self.assertEqual(veto["reason"],"test_only_injection")

    def test_cache_binding_and_fixed_trace_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);prepared,caches,summary=self.prepare(root)
            summary["cache_sha256"]["forward"]="0"*64
            with self.assertRaisesRegex(StrictPairGateError,"cache binding"):
                strict_pair_gate(pair_id=PAIR,arm="sgf_selected_union",prepared_path=prepared,
                    forward_cache_path=caches[0],reverse_cache_path=caches[1],dual_summary=summary,
                    preregistration=SPEC,icp_fn=self.icp,rule_features_fn=self.rule,
                    test_injection=True)
            prepared,caches,summary=self.prepare(root)
            def bad_icp(source,reference,initial,*,seed):
                row=self.icp(source,reference,initial,seed=seed)
                row["trace"][0]["fixed_correspondence_rmse_after_m"]=1.
                return row
            result=strict_pair_gate(pair_id=PAIR,arm="sgf_selected_union",prepared_path=prepared,
                forward_cache_path=caches[0],reverse_cache_path=caches[1],dual_summary=summary,
                preregistration=SPEC,icp_fn=bad_icp,rule_features_fn=self.rule,
                test_injection=True)
            self.assertFalse(result["safe"])

    def test_injected_callables_are_rejected_in_formal_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);prepared,caches,summary=self.prepare(root)
            with self.assertRaisesRegex(StrictPairGateError,"co-sealed|source SHA mismatch"):
                strict_pair_gate(pair_id=PAIR,arm="sgf_selected_union",prepared_path=prepared,
                    forward_cache_path=caches[0],reverse_cache_path=caches[1],dual_summary=summary,
                    preregistration=SPEC,icp_fn=self.icp,rule_features_fn=self.rule)


if __name__=="__main__":unittest.main()
