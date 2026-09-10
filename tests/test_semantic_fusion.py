import unittest
import numpy as np
from pose_pipeline.semantic_fusion import transfer_consensus, transfer_surfel_consensus, transfer_surfel_completion
from pose_pipeline.semantic_reconcile import attach_instances


def observation(probabilities):
    return {'cloud':{'xyz':np.array([[0.,0.,0.]]),'normals':np.array([[0.,0.,1.]]),
                     'labels':np.array([1])},
            'graph':{'node_labels':[1],'semantic_labels':[max(probabilities,key=probabilities.get)],
                     'semantic_confidences':[max(probabilities.values())],
                     'semantic_probabilities':[probabilities]}}


class SemanticFusionTests(unittest.TestCase):
    def fuse(self,observations,maps,point=(0.,0.,0.)):
        return transfer_consensus(np.array([point]),np.array([[0.,0.,1.]]),observations,maps,{'table':1,'floor':2})

    def test_same_class_unresolved_instances_retains_semantics(self):
        s=observation({'table':.9,'floor':.1})
        sem,inst,conf,count=self.fuse([s,s],[{1:10},{1:20}])
        self.assertEqual(sem[0],1);self.assertEqual(inst[0],0)
        self.assertAlmostEqual(float(conf[0]),.9,places=5);self.assertEqual(count,1)

    def test_linked_instances_persist(self):
        s=observation({'table':.9,'floor':.1})
        sem,inst,_,_=self.fuse([s,s],[{1:10},{1:10}])
        self.assertEqual((sem[0],inst[0]),(1,10))

    def test_conflicting_classes_abstain(self):
        a=observation({'table':.9,'floor':.1});b=observation({'table':.1,'floor':.9})
        sem,inst,conf,_=self.fuse([a,b],[{1:10},{1:10}])
        self.assertEqual((sem[0],inst[0],conf[0]),(0,0,0))

    def test_unobserved_geometry_remains_unknown(self):
        sem,inst,conf,_=self.fuse([observation({'table':1.})],[{1:10}],(.1,0.,0.))
        self.assertEqual((sem[0],inst[0],conf[0]),(0,0,0))

    def test_surfel_footprint_covers_surface_but_not_depth_gap(self):
        s=observation({'table':1.});s['cloud']['radius_m']=np.array([.06])
        xyz=np.array([[.06,0.,0.],[0.,0.,.06],[.12,0.,0.]])
        normals=np.tile([0.,0.,1.],(3,1))
        sem,inst,_,_=transfer_surfel_consensus(xyz,normals,[s],[{1:10}],{'table':1,'floor':2})
        np.testing.assert_array_equal(sem,[1,0,0])
        np.testing.assert_array_equal(inst,[10,0,0])

    def test_surfel_footprint_requires_measured_radius(self):
        s=observation({'table':1.})
        with self.assertRaises(KeyError):
            transfer_surfel_consensus(np.zeros((1,3)),np.array([[0.,0.,1.]]),[s],[{1:10}],{'table':1})

    def test_sga_conflict_cannot_erase_or_replace_full_sgf_semantics(self):
        sem=np.array([1,2,1,0]);conf=np.array([.9,.8,.7,0.])
        result,instance,confidence=attach_instances(sem,conf,np.array([2,2,1,1]),np.array([9,8,0,6]))
        np.testing.assert_array_equal(result,sem)
        np.testing.assert_array_equal(confidence,conf)
        np.testing.assert_array_equal(instance,[0,8,0,0])

    def test_disc_completion_never_replaces_existing_center_evidence(self):
        s={'cloud':{'xyz':np.array([[0.,0.,0.],[.05,0.,0.]]),
             'normals':np.array([[0.,0.,1.],[0.,0.,1.]]),'labels':np.array([1,2]),'radius_m':np.array([0.,.08])},
           'graph':{'node_labels':[1,2],'semantic_labels':['table','floor'],
             'semantic_confidences':[1.,1.],'semantic_probabilities':[{'table':1.},{'floor':1.}]}}
        xyz=np.array([[.02,0.,0.],[.10,0.,0.]])
        semantic,_,confidence,_=transfer_surfel_completion(xyz,np.tile([0.,0.,1.],(2,1)),[s],[{1:10,2:20}],{'table':1,'floor':2})
        np.testing.assert_array_equal(semantic,[1,2])
        np.testing.assert_array_equal(confidence,[1.,1.])


if __name__=='__main__':unittest.main()
