# b716 fixed4 execution contract fix3

This commit is a contract-only hardening change.  It does not authorize or run
ColorPCR, PointDSC, PyGCRANSAC, ICP, a model, GPU work, reconstruction, or
refusion.  The checked-in stage entrypoint always exits 78.

## Independent authorization

Production authorization and the independent guard-audit receipt must each be
Ed25519-signed by the public key named by the fixed external trust anchor.  The
anchor path and its exact SHA-256 are code constants, and the anchor and public
key must both be outside the repository and output root.  The private key is
not present in this repository.  The checked-in anchor digest is deliberately
all-zero/unprovisioned and is rejected, so this commit cannot authorize a real
run.  Provisioning requires a later independently reviewed source change.

The signed authorization continues to bind TTL, repository root, Git HEAD and
tree, output root, all 107 task IDs, task manifest, exact191/prepared inputs,
the 6,091-node ownership closure, source closure, subprocess/runtime closure,
and the separately signed guard-audit receipt.

## Non-callable execution boundary

Production execution no longer imports or invokes the legacy Python runner
registry.  Every task contains an exact execution binding for:

- `/usr/bin/dash` and its SHA-256;
- the checked-in disabled shell entrypoint and its SHA-256;
- `/usr/bin/strace`, interpreter shared-library/runtime closure, and hashes;
- argv, environment, cwd, control inputs, output receipt paths, and exit code.

The parent revalidates Git/source/interpreter/runner/runtime closure before and
after the child.  It records create-only stdout, stderr, strace, and a parent
consumption receipt.  Undeclared successful file opens fail closed.  The
trusted parent derives `failure_type` from the observed trace and exit status;
runner output is never interpreted as a safety result.

## Recursive real-file closure

An exact191 artifact accepted by fix3 must contain a non-empty
`recursive_real_file_closure` and its stable digest.  Each closure row names an
absolute root and exhaustively lists every regular file below it with relative
path, byte count, and SHA-256.  Validation rejects symlink roots, symlink files,
path escape, duplicate roots/files, empty or digest-only rows, and omitted or
extra files.  Every declared file is opened with `O_NOFOLLOW` and read to EOF.

## Preserved safety rules

The fix2 evidence mapping, exact20 solver matrix, 34-hypothesis replay,
typed-failure no-transform rule, all-12 known-bad permanent veto, create-only
result/attempt rules, upstream recomputation, GT/official92/threshold/result
selection prohibitions, and reconstruction/refusion artifact rejection remain
in force.  No production authorization is shipped by fix3.
