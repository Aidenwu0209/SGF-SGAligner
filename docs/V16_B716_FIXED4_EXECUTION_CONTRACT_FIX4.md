# b716 fixed4 execution contract fix4

> Migration note (fix5): this document records the pre-trust-anchor-migration
> snapshot at commit `82380deeb491d85418677292874a8a2968c4ddaa`. Its
> all-zero anchor digest and whole-authority-directory prohibition are historical
> evidence, not the current provisioned fix5 contract. See
> `V16_B716_FIXED4_EXECUTION_CONTRACT_FIX5_ANCHOR.md` for the migrated state.

This is a metadata/contract hardening change only. It does not authorize or
run GPU, model, ColorPCR, PointDSC, PyGCRANSAC, ICP, reconstruction, refusion,
GT, or official92 work. The checked-in stage entrypoint remains permanently
disabled and exits 78.

## Immutable trust inputs and off-host signer gate

Security decisions reconstruct the anchor path/digest and the OpenSSL
path/digest from function-local code literals on every verification. No Python
module global carries these values; caller-created replacement attributes are ignored. The
anchor and public key are reopened through every parent component with
`O_NOFOLLOW`; their SHA-256 values are checked again before use. `/usr/bin/openssl`
is also code/SHA pinned.

The production anchor SHA remains deliberately all-zero and is rejected. In
addition, execution refuses authorization while
`/home/aidenwu/.local/share/sgaligner-exact72-audit-authority-v1` exists on the
execution host, and a signed authorization must contain
`signer_private_key_not_on_execution_host=true`. The current host therefore
remains blocked: its signer private-key material is owned by the same
`aidenwu` UID as the execution process. No authorization is created here.

## Fresh-process, SHA-pinned executor

The caller process never invokes a stage Python callable. It launches
`scripts/v16_b716_fixed4_sealed_executor.py` in a fresh isolated Python
process. The exact Python executable, executor source, dash interpreter,
disabled stage runner, and strace executable are all code/SHA pinned. The
fresh process independently reopens the control files and rechecks the fixed
signature boundary before it can invoke the disabled stage shell. Reassigning
a parent module function or registry cannot redirect this entrypoint.

Task, preflight, authorization, and task-manifest paths plus their SHA-256
values are explicit argv fields. The shell validates the fixed argv layout and
reads every control object to EOF. Parent strace evidence must contain all four
successful opens and no undeclared opens; merely observing a disabled shell
exec is insufficient.

## Anchored paths and exact72 recursive closure

Output root, task root, wrapper directory, stdout, stderr, trace, and receipt
are opened or created through root-anchored dirfds with `O_NOFOLLOW` and
create-only leaves. A trace fd is created below the authorized root and passed
to strace, so no temporary trace product is placed elsewhere. Any parent or
leaf symlink, path escape, partial wrapper, or overwrite attempt fails closed.

Each of the five exact72 top-level inputs must be the same absolute file as one
member of the signed exhaustive recursive closure. For every one of the 72
results, `task.json`, `authorized_task_view.json`, `attempt_receipt.json`,
`result.json`, and `correspondences.npz` must occur under the matching task ID
directory in that same closure. Every parent is opened without following
symlinks; semantic task SHA and file SHA bindings are recomputed.

All prior fixed4 prohibitions, exact task/evidence counts, exact20 matrix,
typed-failure replay, known-bad veto, and reconstruction/refusion rejection
remain unchanged and fail closed.
