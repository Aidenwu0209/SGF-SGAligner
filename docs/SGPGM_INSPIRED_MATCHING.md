# SG-PGM-inspired matching extensions

## Status

Research-only, opt-in, disabled by default.

The implementation adds three independent inference controls without changing
the checkpoint or the official top-3 default:

1. `sinkhorn_partial`: balanced assignment followed by a learned match-count
   budget and a discrete one-to-one selection;
2. `P2SG-lite`: a 13-D rigid-invariant object geometry signature fused with
   scene-graph embedding similarity;
3. graph-guided point-correspondence re-ranking before the existing
   PyGCRANSAC/ICP/fail-closed decision path.

`P2SG-lite` is not a claim of faithful KPConv feature pooling. A faithful
P2SG implementation requires propagating point-backbone features and training
a new checkpoint. The current implementation promotes only the behavior that
was directly tested in the shadow ablation.

## Leakage boundary

The adaptive budget weights, geometry normalization statistics and alpha=0.35
were fit on 85 selection89 development pairs. Fixed4 was excluded. At inference
time the code reads only embeddings, object points and predicted graph scores;
it does not read anchors or GT transforms.

Calibration identifier:
`selection89-dev85-fixed4-excluded-20260831`.

## Usage

Historical production-compatible default:

```bash
python -m inference.sgf_official.inference \
  --mode official_sgf_predicted \
  --pair-id <source_uuid>_to_<reference_uuid> \
  --output outputs/default
```

Combined experimental preset:

```bash
python -m inference.sgf_official.inference \
  --mode official_sgf_predicted \
  --pair-id <source_uuid>_to_<reference_uuid> \
  --output outputs/sgpgm_experimental \
  --sgpgm-experimental-preset
```

Independent arms:

```bash
# Partial assignment only
--matching-policy sinkhorn_partial

# Partial assignment plus P2SG-lite geometry
--matching-policy sinkhorn_partial --geometry-fusion-alpha 0.35

# Official top-3 nodes plus graph-guided point re-ranking
--graph-rescore-beta 1.0
```

Every run records the effective settings, calibration identifier and
`gt_at_inference=false` in `node_matches.json`, `registration_result.json` and
`status.json`.

## Verification result and promotion gate

The offline Fixed4 node-matching ablation improved macro F1 from 0.0747
(official cached candidates) to 0.1671 with partial assignment and 0.2580 with
P2SG-lite. Candidate count fell from 146 to 40.

The same-code, same-input Fixed4 A/B run showed an important interaction:

| Pair | Default node F1 | Experimental node F1 | Default RRE/RTE | Experimental RRE/RTE | Default/experimental decision |
|---|---:|---:|---:|---:|---|
| `09582205_1883` | 0.1795 | 0.2400 | 3.48 deg / 0.103 m | 29.10 deg / 0.642 m | accept / reject |
| `68bae76c_5364` | 0.0741 | 0.6000 | 7.62 deg / 0.679 m | 98.28 deg / 0.851 m | reject / reject |
| `6a36052f_c2b5` | 0.1429 | 0.5000 | 88.03 deg / 3.215 m | 27.31 deg / 1.165 m | reject / reject |
| `f38169cf_56fe` | 0.0000 | 0.0000 | 74.23 deg / 0.951 m | 73.68 deg / 0.930 m | reject / reject |

The enhancement improves node F1 on three of four pairs and substantially
reduces pose error on `6a36052f_c2b5`, but it creates no new accepted pair and
regresses the one accepted default pair. The existing gate rejected all four
experimental results. Therefore the combined preset is not promoted to the
default and must not be described as an end-to-end improvement.

Verification also covered 11 focused enhancement/candidate tests, 68 related
matching/adapter/registration tests, and real-data adapter execution. The first
pytest environment lacked `plyfile` for one real-data test; the same 16 adapter
tests were rerun in the SGAligner runtime containing `plyfile` and all passed.

Promotion requires all of the following on sealed same-input runs:

- no regression in official-top3 default outputs;
- improved or non-inferior Fixed4/fixed12 registration and decision metrics;
- no increase in false releases or ambiguity;
- final matched-surface and fused-geometry inspection;
- a separately trained/evaluated checkpoint for a faithful KPConv P2SG path.
