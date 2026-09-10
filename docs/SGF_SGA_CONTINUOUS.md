# Continuous SGF + SGA semantic mapping

Experimental labels over a frozen complete RGB-D trajectory and point cloud.
Recognition uses SceneGraphFusion; cross-stream object matching uses SGAligner.
No OneFormer, ground-truth labels or ground-truth poses enter inference.

1. Replay calibrated RGB-D continuously with `python -m pose_pipeline.semantic_continuous`.
   Required: `--manifest --trajectory --model --output`. Supply the per-scene,
   GT-free `--frame-control`; its matrix must never be copied from another scene.
   `--profile scannet-historical` selects pyr3 / edge0.90 / filter128 and rejects
   non-ScanNet inputs. Default `legacy` retains pyr2 / edge0.98 / filter96.
   Sampling defaults to512, seed42. Full, even and odd streams are available.
   Snapshots preserve native surfel radius in metres, normals, RGB, region IDs,
   semantic probabilities and relations. Intermediate snapshots are cumulative.
2. Run actual SGAligner pct/gat/rel inference on the even/odd final graphs through
   `semantic_mapping.associate(..., t_model_world=H)`, using the same model frame.
   Accept matches only after semantic and measured geometric checks. Consolidate
   native same-part relations with `semantic_instances.consolidate` and save the
   association report and global ID maps. An accepted match never changes a pose.
3. Export with `python -m pose_pipeline.semantic_reconcile --full FULL/final
   --first EVEN/final --second ODD/final --association REPORT.json
   --global-ids IDS.json --classes CLASSES.json --baseline BASELINE.ply
   --output NEW_OUTPUT --sequence SEQUENCE --use-surfel-footprints`.
   The two SGA streams must partition the full stream. Full-stream semantics are
   preserved; measured surfel discs only fill unknown points. Instance ambiguity
   does not erase a class label. Objects distinguish SGA cross-stream support
   from provisional SGF stream-local evidence. All original PLY fields stay equal.

These new entrypoints are opt-in. The existing `run-semantic` command remains
its earlier short-window implementation; do not use it to reproduce this route.
Native SGF and GPU SGA may need separate Python processes and library paths.

## ScanNet0030 development evidence, 2026-09-10

Complete2498-frame semantic replay over develop@1cf90f7 geometry, 462705 vertices.
Same fixed185436 GT vertices,5cm tolerance, offline evaluation only:

| Measure | Earlier semantic route | Restored full SGF + SGA |
|---|---:|---:|
| Geometry coverage |91.51%|91.51%|
| Semantic coverage |11.70%|67.31%|
| Correct-label coverage |0.63%|52.14%|
| Accuracy where labelled |5.38%|77.46%|

51 accepted SGA correspondences;88 exported IDs,37 with accepted cross-stream
support.19 semantic unit tests passed. Geometry/RGB fields and already-known
semantic labels/confidences remained identical. Existing v5 historical semantic
coverage79.31% is higher, with different input conditions; superiority is not
claimed. This one development scene is not multi-dataset quality acceptance.

Evidence bundle: `comparisons/sgf_sga_restore_20260910_v1` in the enclosing
SGF-SGA workspace, including REPORT_zh.md, FINAL_VERIFICATION.json,
REMOTE_FINAL_RECEIPT.json, frozen runtime sources, predictions and final PLYs.
