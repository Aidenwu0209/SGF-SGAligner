# Fixed4 production manifest transaction v2

The production-manifest builder accepts only the active protocol-ready path:

- outer preflight schema `v16-b716-fixed4-execution-preflight-v5` with
  `active_subprocess_contract.schema =
  v16-b716-fixed4-active-subprocess-preflight-v2` and
  `production_adapter_protocol_ready = true`;
- task schema `v16-b716-fixed4-operational-task-v5` with a sealed
  `v16-b716-fixed4-active-stage-input-descriptor-v2` whose readiness flag is
  true; and
- an active `execution_binding` whose `stage_implementation_status` is
  `production_adapter_ready`.

Legacy v1 controls, missing descriptors, false readiness, and fixture/disabled
bindings are rejected before any control output is created.

## Transaction layout

Each attempt receives a new create-only directory:

```
tasks/<task-id>/control/production_manifest_transactions/<tx-id>/
  production_input_manifest.json
  production_execution_manifest.json
  COMMITTED.json
```

`COMMITTED.json` is written last and uses schema
`v16-b716-fixed4-production-manifest-transaction-commit-v2`.  It binds the task
ID and task payload SHA, transaction ID, stage, UTC `created_at`, and both
manifest path/file-SHA/payload-SHA rows.  Consumers must call
`load_committed_production_manifest_transaction` (or apply the same exact
checks) and must not discover either manifest by filename alone.

An interrupted attempt remains uncommitted and therefore invisible.  A retry
uses a new transaction ID; it never overwrites or completes the abandoned
attempt.  The CLI prints a receipt with schema
`v16-b716-fixed4-production-manifest-build-receipt-v2`; its `receipt_path`,
`receipt_sha256`, and `receipt_payload_sha256` identify the final commit marker.

## Dispatcher integration fields

The dispatcher must take these fields from the successful builder receipt,
not reconstruct fixed canonical filenames:

- `transaction_id`
- `transaction_state` (must equal `COMMITTED`)
- `receipt_path`, `receipt_sha256`, `receipt_payload_sha256`
- `production_input_manifest.{path,bytes,sha256,payload_sha256}`
- `production_execution_manifest.{path,bytes,sha256,payload_sha256}`
- `authorization_binding` (including the two transaction manifest bindings)

No authorization request may be signed until the commit marker and both
manifest rows have independently revalidated.
