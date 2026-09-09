import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import tempfile
import numpy as np
from pose_pipeline.semantic_mapping import sgf_pose, windows, transfer_labels, replay_submap
from pose_pipeline.semantic_instances import consolidate


def scene(x, label=1, confidence=.9):
    return {'cloud': {'xyz':np.array([[x,0.,0.]]),
                      'normals':np.array([[0.,0.,1.]]),'labels':np.array([label])},
            'nodes':{label:{'label':'chair','confidence':confidence}}}


class SemanticMappingTests(unittest.TestCase):
    def test_native_disabled_prediction_is_failure_not_successful_empty_labels(self):
        stopped = []
        native = SimpleNamespace(snapshot_inseg=lambda **_: {},
                                 snapshot_graph=lambda **_: {'prediction_enabled':False})
        engine = SimpleNamespace(native=native,process_frame=lambda _:None,
            run_full_prediction=lambda **_:None,stop=lambda:stopped.append(True))
        module = SimpleNamespace(CameraIntrinsics=lambda *args:None,FramePacket=lambda *args:None,
                                 SceneGraphFusion=lambda *args,**kw:engine)
        frame = SimpleNamespace(frame_id=1,timestamp_us=1000)
        pose = SimpleNamespace(t_world_camera=np.eye(4))
        with tempfile.TemporaryDirectory() as folder, patch.dict('sys.modules',{'sgf_runtime':module}), patch(
                'pose_pipeline.semantic_mapping._read_rgbd',return_value=(
                    np.zeros((2,2,3),np.uint8),np.ones((2,2),np.uint16),(1,1,0,0))):
            with self.assertRaisesRegex(RuntimeError,'disabled prediction'):
                replay_submap([(frame,pose)],1000,Path(folder),Path(folder)/'output')
        self.assertEqual(stopped,[True])

    def grouping_scene(self,offset=.02,label='chair',relation_confidence=.9):
        rng = np.random.default_rng(7)
        a = rng.uniform([0,0,0],[.05,.05,.05],(60,3))
        b = a + [offset,0,0]
        return {'cloud':{'xyz':np.concatenate([a,b]),'labels':np.repeat([1,2],60)},
            'nodes':{1:{'label':'chair','confidence':.9,'native_instance_id':1},
                     2:{'label':label,'confidence':.9,'native_instance_id':1}},
            'graph':{'relation_edges':[[1,2]],'relation_labels':['same part'],
                     'relation_confidences':[relation_confidence]}}

    def test_same_part_merges_touching_compatible_fragments(self):
        maps,report = consolidate([self.grouping_scene()],[{1:10,2:20}])
        self.assertEqual(maps,[{1:10,2:10}])
        self.assertEqual(len(report['accepted_merges']),1)

    def test_same_part_rejects_distant_or_semantically_conflicting_objects(self):
        for s in (self.grouping_scene(offset=1),self.grouping_scene(label='table'),
                  self.grouping_scene(relation_confidence=.2)):
            maps,_ = consolidate([s],[{1:10,2:20}])
            self.assertEqual(maps,[{1:10,2:20}])

    def test_same_part_merge_propagates_across_sga_linked_submaps(self):
        s = self.grouping_scene()
        maps,_ = consolidate([s,s],[{1:10,2:20},{1:20,2:30}])
        self.assertEqual(maps,[{1:10,2:10},{1:10,2:10}])

    def test_pose_inverse_rotation_and_mm_translation(self):
        pose = np.array([[0.,-1.,0.,1.],[1.,0.,0.,2.],[0.,0.,1.,3.],[0.,0.,0.,1.]])
        converted = sgf_pose(pose)
        self.assertEqual(converted.dtype,np.float32)
        camera = converted[:3,:3] @ np.array([1000.,2000.,3000.]) + converted[:3,3]
        np.testing.assert_allclose(camera,0)
        np.testing.assert_allclose(converted[:3,3],[-2000.,1000.,-3000.])

    def test_windows_cover_tail_without_redundant_last_window(self):
        self.assertEqual(list(windows(187,120,30)),[(0,120),(90,187)])
        self.assertEqual(list(windows(120,120,30)),[(0,120)])
        with self.assertRaises(ValueError):
            list(windows(100,120,120))

    def transfer(self, scenes, ids):
        return transfer_labels(np.array([[0.,0.,0.]]),np.array([[0.,0.,1.]]),
                               scenes,ids,{'chair':5})

    def test_surface_too_far_stays_unknown(self):
        semantic,instance,_,_ = self.transfer([scene(.06)],[{1:10}])
        self.assertEqual(int(semantic[0]),0)
        self.assertEqual(int(instance[0]),0)

    def test_conflicting_close_instances_abstain(self):
        _,instance,_,ambiguous = self.transfer([scene(.01),scene(.012)],[{1:10},{1:11}])
        self.assertEqual(int(instance[0]),0)
        self.assertEqual(ambiguous,1)

    def test_same_global_instance_does_not_conflict(self):
        semantic,instance,_,ambiguous = self.transfer([scene(.01),scene(.012)],[{1:10},{1:10}])
        self.assertEqual(int(instance[0]),10)
        self.assertEqual(int(semantic[0]),5)
        self.assertEqual(ambiguous,0)

    def test_runner_up_becomes_clear_winner(self):
        _,instance,_,_ = self.transfer([scene(.03),scene(.035),scene(.005)],
                                      [{1:10},{1:11},{1:11}])
        self.assertEqual(int(instance[0]),11)

    def test_low_confidence_and_wrong_normal_stay_unknown(self):
        _,instance,_,_ = self.transfer([scene(.01,confidence=.2)],[{1:10}])
        self.assertEqual(int(instance[0]),0)
        s = scene(.01); s['cloud']['normals'] = np.array([[1.,0.,0.]])
        _,instance,_,_ = self.transfer([s],[{1:10}])
        self.assertEqual(int(instance[0]),0)


if __name__ == '__main__':
    unittest.main()
