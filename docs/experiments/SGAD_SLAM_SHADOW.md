# SGAD-SLAM shadow frontend

This branch reserves SGAD-SLAM for an independent RGB-D shadow run and retains
the existing SGF pose graph/refusion backend.

## Official-source gate result

Locked official commit: `2dca26dcef242edf07c3c361be390d3ca254aa43`.

The current official source does **not** satisfy the SGF no-GT inference
contract:

- `mp_Mapper.py` initializes `estimated_c2ws[0]` from `dataset[0][3]`;
- even with `gt_camera=False`, frames 0 and 1 are assigned dataset poses;
- the ScanNet loader unconditionally reads `gt_pose.txt`;
- the TUM loader unconditionally reads `groundtruth.txt` or `pose.txt`.

Consequently, running the official configuration and reporting its ATE would
cross the GT boundary. `scripts/audit_sgad_no_gt.py` fails with exit code 2 on
this source. GPU feasibility and performance are not promotion evidence until
an isolated provider copy uses identity for frame 0, RGB-D odometry for frame 1,
never opens pose files, and produces a complete audited trajectory.

After that source gate passes, `scripts/import_sgad_shadow.py` accepts only one
finite metric `T_world_camera` per manifest frame. The final decision still
requires same-input SGF refusion and geometry comparison; SGAD's own Gaussian
rendering metrics are not sufficient.
