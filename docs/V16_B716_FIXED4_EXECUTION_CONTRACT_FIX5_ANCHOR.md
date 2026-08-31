# b716 fixed4 execution contract fix5 public trust anchor

This change provisions only an independently stored public verification trust
anchor. It creates no execution authorization and does not run GPU, model,
solver, ColorPCR, PointDSC, PyGCRANSAC, ICP, reconstruction, refusion, GT, or
official92 work. The checked-in stage runner remains disabled and exits 78.

## Code-pinned public trust

`verify_fixed_signed_document` reconstructs these immutable literals inside the
function on every call:

- anchor path:
  `/home/aidenwu/Documents/fixed4-independent-trust-anchor/trust-anchor-v1.json`
- anchor file SHA-256:
  `f490dc70fbcfe7887a50de9f8d50c316b2226eb7c8bba81ddb21d4f3f1efca0b`
- prohibited private-key leaf:
  `/home/aidenwu/.local/share/sgaligner-exact72-audit-authority-v1/audit_private.pem`

The authority directory and `audit_public.pem` may remain on the execution
host. Only presence of the exact prohibited private-key leaf fails closed.

The external trust-anchor directory is mode `0555`; `audit_public.pem`,
`trust-anchor-v1.json`, and `SHA256SUMS` are mode `0444`. The public key SHA-256
is `440c71b363e29457b41a491c2c342e06c7d273be016d2f7697852beed1789b38`.
The canonical anchor payload SHA-256 is
`805b9553cd6a8cadc9c5ea53aeba9c08c70662d645472920ba4a80bacb77fd03`.
The signature algorithm identifier is
`ed25519-openssl-pkeyutl-raw`.

The fix4 document remains as migration-before evidence. Its all-zero digest and
whole-authority-directory prohibition do not describe fix5.

## Residual execution block

Public trust provisioning is not authorization. No signed authorization is
created by this change; all existing authorization, guard-audit, exact task,
evidence, source/runtime closure, GT/official92, reconstruction, and refusion
gates remain fail closed.
