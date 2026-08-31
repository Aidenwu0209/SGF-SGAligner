# V7 ICP trace gate: fixed-correspondence correction

This research branch corrects one measurement bug in the V7 ICP trace gate.
It does **not** change ICP, RANSAC, Rule-B, consensus radii/quorum, policy
ordering, or any formal split gate.

## Root cause

The historical `rmse_before_m` / `rmse_after_m` fields are thresholded
full-surface diagnostics.  The latter recomputes nearest neighbours and the
20 cm inclusion set after the Kabsch update.  A larger value can therefore be
caused by different points entering or leaving the statistic; it is not proof
that Kabsch increased its own objective.

The corrected trace records both views:

- `surface_rmse_*` and `surface_correspondences_*` preserve the old changing
  nearest-neighbour/threshold diagnostic;
- `fixed_correspondence_rmse_*` evaluates the transform before and after the
  Kabsch update on the exact same source/reference pairs;
- `trace_gate` fail-closes when the fixed metric is absent and uses only this
  fixed objective for the protocol's non-increasing-RMSE condition.

The last-update limits remain 0.25 degrees and 0.005 m.  All other frozen
numbers are unchanged.

## Regressions

The synthetic regression has a first-step thresholded surface RMSE increase
from 0.1362084527 m to 0.1400988766 m while the same-step fixed-correspondence
objective decreases from 0.1362084527 m to 0.1324545020 m.  The old gate
rejects this event; the corrected mathematical gate accepts it.

A read-only replay test is available through `V7_FROZEN_WORKER_JSON`.  It
validates the frozen worker hash and cache SHA, reconstructs canonical surfaces,
replays ICP from the frozen raw transform and seed, and requires the final
transform to match within 1e-12.

The audited V7 batch contained 18 pair/outer aggregate instances with a selected
forward medoid (144 policy references, 17 unique frozen worker artifacts).  A
read-only replay of all 17 unique workers found:

- old changing-set surface metric non-monotonic: 17/17;
- corrected fixed-correspondence metric non-monotonic: 0/17;
- reproduced final-transform mismatch: 0/17;
- corrected trace gate usable, including unchanged last-update limit: 17/17.

No formal 12-pair rerun was performed and no historical artifact was modified.

## Evidence boundary

An independent, separate offline ablation has reported that an all-valid-ICP
consensus variant at `r5/t0.1/q4` yields 7 correct and 0 error on both 12-pair
outer repeats.  That candidate changes consensus semantics, is not implemented
or verified by this commit, and must be reviewed in its own branch/protocol
before use.  It must not be conflated with this trace-metric correction.
