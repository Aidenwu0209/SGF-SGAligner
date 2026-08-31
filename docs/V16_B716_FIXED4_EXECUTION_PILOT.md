# b716 fixed4 execution pilot contract

> **Fix2 authority:** the v1 callable-runner procedure below is superseded by
> `V16_B716_FIXED4_EXECUTION_CONTRACT_FIX2.md`. Callers must not pass a runner
> callable. The checked-in runner registry remains execution-disabled, so this
> commit can prepare and verify metadata but cannot launch a task, GPU, model,
> solver, reconstruction, or refusion. Any future executable runner requires a
> new source SHA, Git HEAD/tree binding, independent guard audit, and expiring
> v2 authorization.

## Scope and current authority

This layer turns the existing 6,091-node, execution-disabled fixed4 evidence
DAG into 107 create-only operational jobs without changing any algorithmic
threshold, seed, checkpoint, pair order, or candidate set.  It is intentionally
fail-closed: preparing the jobs does not authorize execution, reconstruction,
refusion, GT use, or official92.

The authoritative order is:

1. exact72 completes under its own sealed authorization;
2. the exact191 merger seals 119 existing plus 72 new rows;
3. the final matched-region builder seals exactly `12/8/2/12` prepared inputs;
4. this module creates 68 directional ColorPCR jobs, 34 bidirectional
   multi-solver pilots, four V16 pair-cluster jobs, and one aggregate job;
5. an independent, expiring authorization receipt binds every input, source,
   task, checkpoint, and frozen rule before a stage-specific runner is called;
6. all 34 hypotheses are replayed.  Typed failures remain explicit and no
   `all_members_ok`, score winner, majority, or result-based selection exists;
7. the known-bad pair replays all 12 hypotheses and is then permanently vetoed;
8. each of the three normal pairs must independently yield one unique,
   compatible, complete-linkage safe-pose cluster.

The operational jobs are coarser than the evidence DAG.  One directional job
contains the two transform sentinels.  One pilot job contains up to eight V14
candidates and the frozen `2 solvers x 2 directions x 5 repeats` matrix, at
most 160 solver rows.  The full evidence receipts must still expand to the
6,091 preregistered nodes.

## Hard P0: registration-defense draft is not authority

The first draft `a8684990f88ad4645ef4ac7d79d617094badca6f` was rejected because
its guard did not reconstruct the attempt evidence and refusion trusted a
recorded decision boolean.  The replacement draft
`662dc38793fdbeb00552812eaf0a213b277bd206` now reconstructs the guard from
the full attempts list, requires two solvers and at least three seeds for each
solver/direction, and makes refusion recompute frozen
`evaluate_registration_decision(features)`.  Its normal and clean tests were
reported green, but its independent audit failed with three P0 defects:
attempt/evidence hashes are not backed by file reads, cross-direction
consensus compares only the first ordered attempt, and decision features are
still caller-forgeable.  Fix3
`d57ad09c1816920f8547f64eae76745a0c258f4c` adds file-backed result,
evidence, receipt, source and decision closures plus a full bipartite
forward/reverse consensus.  It has not been independently audited and its
solver inventory is `colorpcr/pointdsc`, while this frozen V13 pilot uses
`pointdsc/pygcransac`.  It is not an authorized or aligned guard.  Therefore
this pilot sets `reconstruction_authorized=false` and must stop before
refusion.

A consumable guard must bind and consumer-rehash:

- two solvers per direction and at least three unique seeds for every
  solver/direction;
- every attempt receipt SHA-256 and evidence manifest SHA-256;
- the stable `attempts_sha256` over the full attempts list;
- the complete frozen `RegistrationDecision` feature object;
- the frozen `evaluate_registration_decision` source SHA-256;
- a newly recomputed decision exactly equal, field by field, to the recorded
  decision;
- an independent audit receipt binding the guard/refusion source tree and
  normal plus clean-shell tests.

Until that exists, algorithm evidence may be generated after authorization,
but no PLY/refusion write is authorized by this contract.

## Copyable sequence after exact72 closes

Run from the fixed4 contract worktree.  Replace only paths and observed hashes;
do not change thresholds or use a different checkpoint.

```bash
PYTHONPATH=.:src python scripts/v16_b716_exact191_merger.py \
  --candidate-manifest /ABS/candidate_plan/fixed4_manifest.json \
  --candidate-sha256 "$CANDIDATE_SHA" \
  --preflight-manifest /ABS/exact72/execution_preflight.json \
  --preflight-sha256 "$EXACT72_PREFLIGHT_SHA" \
  --preregister /ABS/exact72/preregister.json \
  --preregister-sha256 "$EXACT72_PREREG_SHA" \
  --authorization /ABS/exact72/execution_authorization.json \
  --authorization-sha256 "$EXACT72_AUTH_SHA" \
  --batch-result /ABS/exact72/batch_result.json \
  --batch-result-sha256 "$EXACT72_BATCH_SHA" \
  --output-root /ABS/exact191
```

Use the CLI help in the checked-out exact191 merger as the authority if the
older path names differ.  Then build the 34 prepared inputs:

```bash
PYTHONPATH=.:src python scripts/v16_b716_matched_region_prepared_builder.py \
  --exact191-manifest /ABS/exact191/exact191_manifest.json \
  --exact191-manifest-sha256 "$EXACT191_SHA" \
  --output-root /ABS/prepared34
```

Finally create the disabled operational envelope:

```bash
PYTHONPATH=.:src python scripts/v16_b716_fixed4_execution_pilot.py prepare \
  --exact191-manifest /ABS/exact191/exact191_manifest.json \
  --exact191-manifest-sha256 "$EXACT191_SHA" \
  --prepared-manifest /ABS/prepared34/builder_manifest.json \
  --prepared-manifest-sha256 "$PREPARED34_SHA" \
  --output-root /ABS/fixed4_pilot
```

The expected prepare result is:

- `status=PREPARED_EXECUTION_DISABLED`;
- operational counts `68/34/4/1`, total 107;
- full evidence DAG node count 6,091;
- `execution_authorized=false`;
- `reconstruction_authorized=false`;
- registration-defense status
  `P0_UNAUTHORIZED_PENDING_AUDIT_FIX3_D57AD09C`.

An independent reviewer creates (not this command) an expiring authorization
receipt with schema `v16-b716-fixed4-execution-authorization-v1`.  Verify it:

```bash
PYTHONPATH=.:src python scripts/v16_b716_fixed4_execution_pilot.py \
  verify-authorization \
  --preflight /ABS/fixed4_pilot/execution_preflight.json \
  --preflight-sha256 "$PREFLIGHT_FILE_SHA" \
  --authorization /ABS/fixed4_pilot/execution_authorization.json \
  --authorization-sha256 "$AUTH_FILE_SHA"
```

Only after that succeeds may a reviewed stage-specific runner call
`execute_authorized_task`.  The safety API validates the canonical task path,
the entire create-only task manifest, the authorization and source closures,
and all upstream task/result/attempt receipts before invoking the runner.

## Stop checks

Stop without executing if any of these is true:

- exact72 has fewer than 72 authorized result rows or any input file SHA drifts;
- exact191 is not `119+72=191` or the fixed distribution is not `12/8/2/12`;
- any prepared NPZ path, size, or SHA fails to match the builder manifest;
- the independent authorization is absent, expired, or lacks a required review;
- the clean GPU identity or PointDSC dependency receipt is absent;
- a source, task, result, attempt, or upstream receipt changes;
- any result consumes GT, runs official92, changes a threshold/checkpoint, or
  selects a winner by result;
- a typed failure has a transform, any hypothesis is silently filtered, or the
  known-bad veto is weakened;
- refusion is requested before the registration-defense v2 P0 is closed.
