# Multi-model replacement validation

This feature branch adds the executable boundary around the five proposed
systems. It does **not** claim that any heavyweight checkpoint has already run
or passed the promotion gate. DPV-SLAM and `official_top3` remain the defaults.

## Roles and hard boundaries

| System | Imported official artifact | Allowed role | Forbidden shortcut |
|---|---|---|---|
| ABot-Recon | `camera_poses_noloop.npy`, `camera_poses_loop.npy`, `loop_edges.json` | full-frame frontend or undecided sparse revisit proposals | mixing official-loop and no-loop arms; unscaled trajectories |
| SLAM-Former | `final_traj.txt` TUM keyframe trajectory | anchor corrections propagated over the complete DPV trajectory | identity or interpolated pose fabrication presented as model output |
| MapAnything | overlapping NPZ windows with `frame_ids` and `camera_poses` | non-blocking 8/16-frame background revision | accepting a window whose shared frames fail the SE(3) consistency gate |
| MipMap | PLY, trajectory/COLMAP export, runtime | external offline geometry control | entering the online source tree or being ranked as an online winner |
| FixAnything | video produced from the fixed 61-frame render | presentation only | feeding generated RGB back to pose, geometry, semantics, or scene graph |

Every open-source model runs in its own official environment. The SGF-SGAligner
side imports only official output artifacts, requires the exact model commit and
checkpoint SHA-256, and writes create-only signed JSON contracts.

## Output contracts

- `pose_trajectory.v1`: complete finite metric `T_world_camera` for continuous
  frontends; no identity gap filling.
- `model_runtime_report.v1`: input/checkpoint hashes, resolution, P50/P95,
  throughput, peak VRAM, queue peak, drops and coverage.
- `trajectory_revision.v1`: binds parent/revised trajectory hashes. Pose display
  may update immediately, but the map cannot switch before complete RGB-D TSDF
  refusion.
- `sparse_constraint_proposals.v1`: ABot revisit edges remain
  `pending_registration_decision`; they cannot bypass the existing decision and
  correction audit.
- `external_artifact_manifest.v1`: makes MipMap/FixAnything roles machine
  checkable and prevents presentation outputs from re-entering scientific data.

The frozen protocol is in
`configs/pose/model_replacement_validation.yaml`.

## Adapter commands

First create the existing RGB-D manifest and baseline DPV trajectory. Scale is
never assumed. Supply a robust paired-depth NPZ (`predicted_depth`,
`sensor_depth_m`, optional `confidence`) or a precomputed scale with a named
method.

```bash
python -m pose_pipeline abot-scale-evidence \
  --manifest manifest.json \
  --local-points abot_output/local_points.pt \
  --confidence abot_output/confidence.pt \
  --output abot_scale_pairs.npz

python -m pose_pipeline adapt-abot \
  --manifest manifest.json \
  --poses abot_output/camera_poses_noloop.npy \
  --mode noloop \
  --scale-evidence abot_scale_pairs.npz \
  --model-commit 195cb9240ffc6300e008d2b70e54d281dd7caf4b \
  --checkpoint-sha256 "$ABOT_CHECKPOINT_SHA256" \
  --output abot_noloop.trajectory.json

python -m pose_pipeline import-abot-loops \
  --manifest manifest.json \
  --loop-edges abot_output/loop_edges.json \
  --scale-evidence abot_scale_pairs.npz \
  --output abot_loop_proposals.json

python -m pose_pipeline adapt-slamformer \
  --manifest manifest.json \
  --baseline-trajectory dpv.trajectory.json \
  --final-traj slamformer_output/final_traj.txt \
  --identifier-mode frame_id \
  --model-variant 'V1.1-long@224' \
  --scale-evidence slamformer_scale_pairs.npz \
  --model-commit 0071ca9e6c53aec55572a5557c5fcf3a23cdba5d \
  --checkpoint-sha256 "$SLAMFORMER_CHECKPOINT_SHA256" \
  --output slamformer_revision.trajectory.json

python -m pose_pipeline adapt-mapanything \
  --manifest manifest.json \
  --baseline-trajectory dpv.trajectory.json \
  --windows mapanything/window_*.npz \
  --input-mode conditioned_on_dpv_pose \
  --window-size 8 \
  --metric-scale 1.0 --scale-method metric_depth_conditioned \
  --model-commit 3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9 \
  --checkpoint-sha256 "$MAPANYTHING_CHECKPOINT_SHA256" \
  --output mapanything_revision.trajectory.json
```

The MapAnything exporter must retain the original manifest frame IDs. Its NPZ
contract is:

```python
np.savez(path, frame_ids=np.asarray(frame_ids), camera_poses=c2w_opencv)
```

## Runtime, revision and refusion

Write latency samples as a JSON list in milliseconds, then create the runtime
report and bind a revision. `--affected-frame-id` must cover every changed pose.

```bash
python -m pose_pipeline model-runtime-report \
  --manifest manifest.json --model MapAnything --model-commit COMMIT \
  --checkpoint checkpoint.pth --width 640 --height 480 \
  --latency-ms latency_ms.json --peak-gpu-memory-mb 12000 \
  --output-pose-count 41 --wall-time 3.2 --output runtime.json

python -m pose_pipeline trajectory-revision \
  --parent dpv.trajectory.json --revised candidate.trajectory.json \
  --source MapAnything --affected-frame-id 0 --affected-frame-id 1 \
  --runtime-report runtime.json --output revision.json

python -m pose_pipeline refuse \
  --manifest manifest.json --trajectory candidate.trajectory.json \
  --output final_refusion
```

The trajectory revision is not permission to replace the current TSDF. Only a
successful complete `refuse` artifact may be displayed as the final map.

## MipMap and FixAnything

MipMap remains a manually installed external trial. Use exactly the frozen RGB
frame list, export PLY plus trajectory/COLMAP when available, and record it with
`external-artifact --system MipMap --role offline_geometry_control`.

FixAnything must start from the fixed 61-frame render and is recorded with
`external-artifact --system FixAnything --role presentation_only`. The contract
rejects any other role or frame count. Anchor preservation, temporal consistency
and COLMAP recovery are evaluation measurements; even a passing video remains a
presentation artifact.

## Decision point

Each candidate summary supplies one precomputed quality score and, for online
roles, the full promotion fields consumed by `model-comparison`. The output has
three role rankings and deliberately sets `winner` to `null`. Runtime is shown
but is not a first-round elimination gate. The user chooses an online candidate
only after reviewing the metrics, failure index, fixed views and actual PLYs.
