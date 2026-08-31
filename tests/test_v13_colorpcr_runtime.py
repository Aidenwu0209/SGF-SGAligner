from pathlib import Path
import ast, hashlib, subprocess, tempfile, unittest
import numpy as np

import v13_colorpcr_official_worker as worker
from v13_colorpcr_sentinel_subprocess import compare_sentinels

class RuntimeTests(unittest.TestCase):
    def write(self,path,offset=0.0):
        p=np.array([[0,0,0],[1,2,3]],np.float32);s=np.array([0.8,0.4],np.float32)
        np.savez(path,ref_corr_points=p,src_corr_points=p,corr_scores=s,
                 estimated_transform=np.eye(4)+offset)
    def test_nonzero_sentinel_is_proper(self):
        t=worker.sentinel("proper_nonzero")
        self.assertGreater(np.linalg.norm(t[:3,3]),0)
        self.assertAlmostEqual(np.linalg.det(t[:3,:3]),1.0,places=6)
        self.assertFalse(np.allclose(t,np.eye(4)))
    def test_coarsest_budget_allocation_is_exact_and_deterministic(self):
        self.assertTrue(np.array_equal(worker.proportional_targets([700,500]),[299,213]))
        self.assertEqual(int(worker.proportional_targets([700,500]).sum()),512)
        self.assertTrue(np.array_equal(worker.proportional_targets([3,4]),[3,4]))
    def test_python_tree_hash_ignores_pycache_but_binds_paths_and_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/"geotransformer").mkdir();(root/"experiments/ColorPCR").mkdir(parents=True)
            (root/"geotransformer/a.py").write_text("x=1\n")
            first=worker.python_tree_hash(root)
            (root/"geotransformer/__pycache__").mkdir();(root/"geotransformer/__pycache__/a.py").write_text("junk")
            self.assertEqual(first,worker.python_tree_hash(root))
            (root/"geotransformer/a.py").write_text("x=2\n")
            self.assertNotEqual(first,worker.python_tree_hash(root))
    def test_comparison_accepts_invariant_and_rejects_dependency(self):
        with tempfile.TemporaryDirectory() as t:
            a,b=Path(t)/"a.npz",Path(t)/"b.npz";self.write(a);self.write(b)
            self.assertTrue(all(x["invariant"] for x in compare_sentinels(a,b).values()))
            self.write(b,1e-3)
            with self.assertRaisesRegex(RuntimeError,"sentinel influenced"):compare_sentinels(a,b)
    def test_worker_omits_forbidden_outputs(self):
        source=Path(worker.__file__).read_text();tree=ast.parse(source)
        self.assertNotIn("load_gt_transform",source)
        self.assertIn("gt_node_corr_indices",source)  # only explicit omit list
        self.assertNotIn("output_dict['gt_node_corr_indices']",source)
        self.assertIn('COARSEST_CAP = 512',source)
        self.assertIn('INPUT_VOXEL_M = 0.10',source)
        self.assertNotIn("registration_collate_fn_stack_mode",source)

    def test_tracked_diff_hash_rejects_staged_and_unstaged_edits(self):
        empty=hashlib.sha256(b"").hexdigest()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);subprocess.run(["git","init","-q",str(root)],check=True)
            subprocess.run(["git","-C",str(root),"config","user.email","v13@example.invalid"],check=True)
            subprocess.run(["git","-C",str(root),"config","user.name","V13 test"],check=True)
            file=root/"tracked.txt";file.write_text("sealed\n")
            subprocess.run(["git","-C",str(root),"add","tracked.txt"],check=True)
            subprocess.run(["git","-C",str(root),"commit","-qm","sealed"],check=True)
            self.assertEqual(worker.tracked_diff_hash(root),empty)
            file.write_text("unstaged\n")
            self.assertNotEqual(worker.tracked_diff_hash(root),empty)
            subprocess.run(["git","-C",str(root),"add","tracked.txt"],check=True)
            self.assertNotEqual(worker.tracked_diff_hash(root),empty)

if __name__=="__main__":unittest.main()
