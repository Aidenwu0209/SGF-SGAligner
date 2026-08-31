"""Fix3 stage 2: REAL PCT-freeze behavior tests — instantiate the
official model, run optimizer steps, compare tensors (no source-string
checks)."""
from __future__ import annotations

import copy
import unittest

import numpy as np
import torch

from aligner.sg_aligner import MultiModalEncoder
from safety.cross_graph_loss import cross_graph_infonce

FROZEN = ("object_encoder", "object_embedding")


def build_frozen_model(seed=7):
    torch.manual_seed(seed)
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41, attr_dim=164,
    )
    for name, p in model.named_parameters():
        p.requires_grad = not name.startswith(FROZEN)
    model.object_encoder.eval()
    return model


def synth_batch(n_src=6, n_ref=7, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    n = n_src + n_ref
    # official GAT slices edges per graph and uses LOCAL indices in
    # each graph's own rel_pose matrix
    edges_s = torch.stack([
        torch.randint(0, n_src, (10,), generator=g),
        torch.randint(0, n_src, (10,), generator=g)], dim=1)
    edges_r = torch.stack([
        torch.randint(0, n_ref, (8,), generator=g),
        torch.randint(0, n_ref, (8,), generator=g)], dim=1)
    batch = {
        "tot_obj_pts": torch.randn(n, 512, 3, generator=g).to(device),
        "tot_bow_vec_object_edge_feats": torch.randn(
            n, 41, generator=g).to(device),
        "tot_rel_pose": torch.randn(n, 3, generator=g).to(device),
        "edges": torch.cat([edges_s, edges_r]).to(device),
        "graph_per_obj_count": [np.asarray([n_src, n_ref])],
        "graph_per_edge_count": [np.asarray([10, 8])],
        "batch_size": 1,
        "tot_bow_vec_object_attr_feats": torch.zeros(
            n, 164, device=device),
    }
    positives = torch.zeros(n_src, n_ref, dtype=torch.bool)
    weights = torch.zeros(n_src, n_ref)
    for i in range(n_src):
        j = i % n_ref
        positives[i, j] = True
        weights[i, j] = 0.8
    return batch, positives.to(device), weights.to(device)


def step(model, optimizer, device="cpu", seed=0):
    batch, positives, weights = synth_batch(device=device, seed=seed)
    model.train()
    model.object_encoder.eval()  # protocol: keep encoder in eval
    out = model(batch)
    loss, _ = cross_graph_infonce(
        out["joint"][:batch["graph_per_obj_count"][0][0]],
        out["joint"][batch["graph_per_obj_count"][0][0]:],
        positives, weights,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss)


def snapshot(model):
    snap = {}
    for name, t in list(model.named_parameters()) + list(
        model.named_buffers()
    ):
        snap[name] = t.detach().clone()
    return snap


class TestPCTFreezeBehavior(unittest.TestCase):
    def test_frozen_tensors_unchanged_after_steps(self):
        model = build_frozen_model()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        before = snapshot(model)
        for s in range(3):
            step(model, optimizer, seed=s)
        after = snapshot(model)
        changed_frozen = [
            k for k in before
            if k.startswith(FROZEN)
            and not torch.equal(before[k], after[k])
        ]
        self.assertEqual(changed_frozen, [])

    def test_bn_buffers_unchanged_in_train_mode(self):
        model = build_frozen_model()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        bn_before = {
            k: v.clone() for k, v in model.named_buffers()
            if "running" in k or "num_batches" in k
        }
        model.train()
        model.object_encoder.eval()
        step(model, optimizer)
        bn_after = {
            k: v for k, v in model.named_buffers()
            if "running" in k or "num_batches" in k
        }
        for k in bn_before:
            self.assertTrue(torch.equal(bn_before[k], bn_after[k]), k)

    def test_trainable_modules_do_update(self):
        model = build_frozen_model()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2
        )
        before = snapshot(model)
        stepped = False
        for s in range(5):
            step(model, optimizer, seed=s)
        after = snapshot(model)
        changed_trainable = [
            k for k in before
            if not k.startswith(FROZEN)
            and not torch.equal(before[k], after[k])
        ]
        self.assertTrue(changed_trainable)
        # at least one GAT/relation/fusion parameter moved
        joined = " ".join(changed_trainable)
        for expected in ("structure_encoder", "meta_embedding_rel",
                         "fusion"):
            self.assertIn(expected, joined)

    def test_frozen_state_survives_state_dict_roundtrip(self):
        model = build_frozen_model()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        step(model, optimizer)
        state = copy.deepcopy(model.state_dict())
        # reload into a fresh frozen model (resume path)
        model2 = build_frozen_model(seed=99)
        model2.load_state_dict(state)
        optimizer2 = torch.optim.Adam(
            [p for p in model2.parameters() if p.requires_grad], lr=1e-3
        )
        before = snapshot(model2)
        step(model2, optimizer2, seed=10)
        after = snapshot(model2)
        changed_frozen = [
            k for k in before
            if k.startswith(FROZEN)
            and not torch.equal(before[k], after[k])
        ]
        self.assertEqual(changed_frozen, [])


if __name__ == "__main__":
    unittest.main()
