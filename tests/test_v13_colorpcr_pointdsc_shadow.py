from pathlib import Path
import json, tempfile, unittest
import numpy as np
from safety.v13_colorpcr_pointdsc_shadow import (
    ARMS, DependencyNotAudited, Q4, V13ContractError, arm_arrays, build_color_pair,
    color_preserving_voxel_aggregate,
    independent_solver_q4, load_raw_inseg, validate_solver_worker, worker_plan)

class Tests(unittest.TestCase):
    def raw(self, path, conflict=False):
        xyz=np.array([[0,0,0],[1.0004,0,0],[1.0004,0,0],[0,1,0]],np.float32)
        labels=np.array([0,2,3 if conflict else 2,4],np.int32)
        colors=np.array([[1,2,3,255],[4,5,6,255],
                         [9,9,9,255] if conflict else [4,5,6,255],[7,8,9,255]],np.uint8)
        np.savez(path,xyz=xyz,labels=labels,colors=colors)
    def shadow(self,path):
        p=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32); c=p[:2]; s=np.ones(2,np.float32)
        np.savez(path,source_points=p,reference_points=p,forward_src_corr=c,
                 forward_ref_corr=c,forward_scores=s,reverse_src_corr=c,
                 reverse_ref_corr=c,reverse_scores=s)
    def test_rows_dedup_arms(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"a.npz";self.raw(p);c=load_raw_inseg(p)
            self.assertEqual((len(c.xyz),c.duplicate_rows_removed),(3,1))
            self.assertEqual(len(arm_arrays(c,"fullscan")["xyz"]),3)
            with self.assertRaisesRegex(V13ContractError,"requires sealed"):
                arm_arrays(c,"sgf_selected_union")
            selected=arm_arrays(c,"sgf_selected_union",np.array([[1,0,0],[0,1,0],[0,0,0]],np.float32))
            self.assertEqual(len(selected["xyz"]),3)
            self.assertIn("sealed_v113_shadow",str(selected["membership_json"].item()))
    def test_conflict_fails(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"a.npz";self.raw(p,True)
            with self.assertRaisesRegex(V13ContractError,"conflicting"):load_raw_inseg(p)
    def test_selected_union_grid_ambiguity_fails_closed(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"a.npz"
            xyz=np.array([[0,0,0],[.0004,0,0],[1,0,0],[0,1,0]],np.float32)
            np.savez(p,xyz=xyz,labels=np.ones(4,np.int32),colors=np.ones((4,3),np.uint8))
            cloud=load_raw_inseg(p)
            with self.assertRaisesRegex(V13ContractError,"ambiguous=1"):
                arm_arrays(cloud,"sgf_selected_union",np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32))
    def test_builder(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);self.raw(r/"s.npz");self.raw(r/"r.npz");self.shadow(r/"h.npz")
            m=build_color_pair("s_to_r",r/"s.npz",r/"r.npz",r/"h.npz",r/"o.npz")
            self.assertFalse(m["rescale_applied"]);self.assertEqual(set(m["arms"]),set(ARMS))
            sealed=(r/"o.npz").read_bytes()
            with self.assertRaises(FileExistsError):
                build_color_pair("s_to_r",r/"s.npz",r/"r.npz",r/"h.npz",r/"o.npz")
            self.assertEqual((r/"o.npz").read_bytes(),sealed)
            with np.load(r/"o.npz") as d:
                self.assertEqual(d["fullscan_source_colors"].dtype,np.uint8)
                manifest=json.loads(str(d["manifest_json"].item()))
                self.assertIn("v113_shadow",manifest)
                self.assertEqual(set(manifest["arms"]),{"sgf_selected_union","fullscan"})
                self.assertIn("fullscan_source_voxel10_source_offsets",d.files)
                self.assertEqual(manifest["colorpcr_input_voxel"]["size_m"],0.10)
                self.assertEqual(manifest["colorpcr_input_voxel"]["labels"],
                                 "filter_before_voxel_no_majority_or_solver_input")
    def test_voxel_is_deterministic_mean_and_has_row_provenance(self):
        values={"xyz":np.array([[.049,0,0],[.001,0,0],[.101,0,0]],np.float32),
                "colors":np.array([[30,60,90],[10,20,30],[100,120,140]],np.uint8),
                "source_row_indices":np.array([9,4,12],np.int64),
                "labels":np.array([99,1,7],np.int32)}
        first=color_preserving_voxel_aggregate(values)
        order=np.array([2,0,1])
        second=color_preserving_voxel_aggregate({k:v[order] for k,v in values.items()})
        for key in first:self.assertTrue(np.array_equal(first[key],second[key]),key)
        self.assertTrue(np.allclose(first["xyz"][:,0],[.025,.101]))
        self.assertTrue(np.allclose(first["colors_mean_0_255"][0],[20,40,60]))
        self.assertTrue(np.array_equal(first["source_offsets"],[0,2,3]))
        self.assertNotIn("labels",first)
    def test_plan_160(self):
        self.assertEqual(len(worker_plan(["a_to_b"]*4)),160)
    def test_dependencies_fail_closed(self):
        with self.assertRaises(DependencyNotAudited):validate_solver_worker({"solver":"pointdsc"})
        row={"solver":"pointdsc","dependency_audited":True,"implementation_sha256":"a",
             "checkpoint_sha256":"b","executed":True,"fallback_used":True,
             "correspondence_sha256":"c","evidence_sha256":"d","transform":np.eye(4)}
        with self.assertRaisesRegex(V13ContractError,"fallback"):validate_solver_worker(row)
    def test_consensus_and_veto(self):
        rows=[]
        for s in ("pointdsc","pygcransac"):
            for d in ("forward","reverse"):
                for i in range(Q4.repeats):rows.append({"solver":s,"direction":d,"repeat":i,
                    "dependency_audited":True,"implementation_sha256":"a","checkpoint_sha256":"b",
                    "executed":True,"fallback_used":False,"correspondence_sha256":"c",
                    "evidence_sha256":"d","transform":np.eye(4),"rule_b_safe":True,"known_bad":False})
        self.assertTrue(independent_solver_q4(rows)["safe"]);rows[0]["known_bad"]=True
        self.assertEqual(independent_solver_q4(rows)["reason"],"known_bad_veto")

if __name__=="__main__":unittest.main()
