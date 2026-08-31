# V8 legacy read-only compatibility note

This note is deliberately separate from the sealed
`V8_STAGE_ORDER_CONSENSUS_PROTOCOL.md`. The worker-bound protocol is an
immutable input and must retain SHA-256
`0ab1942f6892295163067a2289fdfd97964e18b2f6ee6959b4e49bccceb2facc`.

One already-frozen V7 development batch predates the later verifier fields
`evidence_mode` and `formal_preregistered`. It may enter a separate read-only
compatibility validator only when its batch file SHA, embedded evidence SHA,
source-snapshot SHA and source commit match the constants recorded in the V8
runner. That validator walks the complete manifest -> batch -> pair receipt ->
aggregate -> worker chain and recomputes transform and permutation hashes. It
never fills the missing fields, rewrites a receipt, or upgrades the evidence;
the output is marked `KNOWN_SUPERSEDED_V7_READ_ONLY`, development-only and
non-formal. Any other old-shaped receipt fails closed.

This compatibility statement does not alter the sealed V8 algorithm, worker
contract, stage order, quorum, thresholds, or eligibility of historical data.
