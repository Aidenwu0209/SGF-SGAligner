# V6-Fix protocol deviations

This file is append-only evidence. It does not amend or relax the committed
pre-registration in `outputs/official_sgaligner_v6_fix_consistency_audit_20260829/`.

## DEV-001: pre-Gate-1 fixed12 execution-chain smoke

- Classification: `PROTOCOL_DEVIATION_SMOKE_ONLY`
- Occurred: 2026-08-29 23:39:37 +0800
- Cause: the implementation owner ran one real fixed12 pair to validate the
  new runner's end-to-end execution and evidence writing before Gate 1 had
  completed.
- Scope: checkpoint A, one pair only, argument `--repeat 90 --limit 1`.
- Result: F/C0/C1 all completed but were non-strict, non-relaxed and rejected.
- Result SHA-256:
  `f16bb4c9b382d152dd4e771cbc94c7547964833fa1dda937219a67a7d31d887e`
- Cache SHA-256:
  `f3cddad406d6298a73048ba3d393fc62859136962fb89003a124492f25ccd2c6`
- Containment: the files remain immutable under the original diagnostic
  directory. They are excluded from every gate, aggregate, checkpoint/path
  selection and Boss metric. They must never be described as a pristine or
  first frozen fixed12 evaluation.
- Consequence: even if later amended runs pass, this stage can be packaged at
  most as `BOSS_RESEARCH_PREVIEW_ONLY`; it cannot claim a clean pre-registered
  `READY_FOR_BOSS_RC` decision.

## DEV-002: diagnostic v1 runner was not a valid gate runner

- Classification: `DIAGNOSTIC_ONLY_IMPLEMENTATION_INVALIDATED`
- Occurred: selection A repeat 00 completed at 2026-08-29 23:51:58 +0800.
- Result SHA-256:
  `00f5563df0e0f1a31510057dc7da6c9e8fdf49bdd72e06324b4ba5d5c09dcae1`
- Observed counts: F 9 strict / 8 accepted-correct; C0 2 / 1; C1 13 / 8
  with one accepted-error; two pairs failed in each path.
- Invalidating findings: F/C0 candidate order was sorted before RANSAC, C1
  used a six-metric mean rather than the pre-registered lexicographic rank,
  and the inference cache key lacked full registration-surface and code
  provenance.
- Containment: the v1 selection output and cache are diagnostic only. They are
  excluded from all gates and are retained solely to explain the first
  divergence and prove why the implementation was corrected.
