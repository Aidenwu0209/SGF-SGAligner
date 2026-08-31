# V16 b716 fixed4 final-orchestrator contract

## Status

`PLANNED_DISABLED_SYNTHETIC_ONLY`.  This commit is not an authorization to run
ColorPCR, a GPU, PointDSC, pygcransac, ICP, official92, or reconstruction.

The contract is based on hardened exact191 commit
`86b2077e06db5a5b1b8e7a2e856ac84c2a89383e` and the prepared-builder schema
finalized by `52c07c471b3a437a020c4975c6ad10777630ef5e`.  The final real manifest
paths and SHAs do not yet exist as reviewed bindings.  They are explicit P0
fields in the preregistration, so authorization remains false.

## Frozen full DAG

The input is exactly 34 hypotheses in fixed pair order `12/8/2/12`:

| Stage | Exact planned rows |
|---|---:|
| prepared inputs | 34 |
| isolated ColorPCR workers (`2 directions x 2 sentinels`) | 136 |
| sentinel-invariant directional caches | 68 |
| exact-three directional caches | 68 |
| V14 candidate sets | 34 |
| reserved V13 solver rows (`8 candidates x 2 x 2 x 5`) | 5,440 |
| strict V13 candidate gates | 272 |
| V15 within-hypothesis complete-linkage decisions | 34 |
| V16 across-hypothesis complete-linkage decisions | 4 |
| final primary-only fixed4 aggregate | 1 |

Total: 6,091 deterministic planned nodes.  A missing V14 candidate slot is an
explicit `typed_not_generated_no_transform`, never an identity transform.

## Failure-member rule

The exact191 allowlists expose 16 frozen typed-failure candidate members and 8
hypotheses containing at least one such member.  The orchestrator cross-binds
those allowlists to all 34 prepared inputs.  Every affected hypothesis is still
present in every planned downstream stage with
`typed_failure_policy=explicit_replay_never_filter`.  Filtering by
`all_members_ok` is forbidden.

Typed failure is evidence about one member, not authority to erase the frozen
surface hypothesis.  Any future stage unable to produce evidence records a
typed failure with no transform.

## Decision rule

- No result selection, best score, largest-cluster choice, or majority vote.
- V13 retains the frozen `2 solvers x 2 directions x 5 repeats`, quorum 4,
  strict ICP trace, and unchanged Rule-B.
- V15 may accept a hypothesis only when all safe candidate realizations form
  one complete-linkage pose cluster under `5 degrees / 0.10 m`.
- V16 may accept a normal pair only when all safe hypotheses form one unique
  complete-linkage pose cluster under the same frozen limits.
- Each of the three normal pairs must satisfy that rule independently.
- The known-bad pair replays all 12 hypotheses and is then permanently vetoed.
- Control cannot rescue primary.

The deterministic medoid fields in existing V15/V16 evidence are descriptive
serialization of an already unique cluster; they are not a path for choosing
between clusters.  A second incompatible safe cluster rejects the row.

## Frozen runtime and resources

- official SGAligner checkpoint SHA: `b716c7d8...a17e`;
- ColorPCR worker seed 7351, neighbor limits `38,36,36,38`, voxel 0.10 m,
  coarsest cap 512;
- existing V13 resource ceiling 160 workers (the DAG is a count, not a
  concurrency request);
- V14 max 8 candidates, max 1,000/min 40 correspondences, residual 0.10 m;
- V13 five repeats, quorum four; ICP seeds 42/43;
- pose compatibility 5 degrees and 0.10 m;
- source/checkpoint hashes are frozen in
  `manifests/v16_b716_fixed4_orchestrator_preregister.json`.

No GT, official92, selection/evaluation labels, posthoc, fallback, threshold
change, or default-checkpoint replacement is allowed.

## Create-only receipts and resume

Each of the 6,091 nodes receives a `planned_disabled` JSON receipt using
exclusive create.  Resume accepts only a byte-semantically identical receipt
whose task identity, upstream list, node SHA, DAG SHA, and preregistration SHA
all match.  A modified or partial receipt fails closed; receipts contain no
model or solver outputs.

Synthetic contract test command:

```bash
PYTHONPATH=.:src python scripts/v16_b716_fixed4_orchestrator_contract.py \
  --preregister manifests/v16_b716_fixed4_orchestrator_preregister.json \
  --output-root /tmp/v16-b716-fixed4-contract \
  --synthetic-fixture
```

The CLI deliberately refuses real manifests in this commit.  A later commit
must bind reviewed real exact191 and prepared-builder paths/SHAs, preserve all
P0 gates, and obtain a separate independent execution authorization.
