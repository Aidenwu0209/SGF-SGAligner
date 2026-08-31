#!/usr/bin/env python3
"""Materialize only the disabled synthetic fixed4 planning contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safety.v13_dual_solver_runtime import atomic_json, sha256_file
from safety.v16_b716_fixed4_orchestrator_contract import (
    build_task_dag,
    materialize_planning_receipts,
    synthetic_fixture_bindings,
    validate_preregister,
    verify_source_pins,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--synthetic-fixture", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_fixture:
        raise SystemExit(
            "real evidence planning is P0-blocked; --synthetic-fixture is required")
    repo = Path(__file__).resolve().parents[1]
    preregister_path = args.preregister.resolve()
    preregister = json.loads(preregister_path.read_text())
    validate_preregister(preregister)
    verify_source_pins(repo, preregister)
    preregister_sha = sha256_file(preregister_path)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dag = build_task_dag(
        synthetic_fixture_bindings(), preregister_sha,
        synthetic_fixture=True)
    dag_path = output_root / "orchestrator_dag.json"
    if dag_path.exists():
        observed = json.loads(dag_path.read_text())
        if observed != dag:
            raise SystemExit("existing DAG differs from frozen synthetic plan")
    else:
        atomic_json(dag_path, dag)
    receipts = materialize_planning_receipts(
        output_root, dag, preregister_sha)
    manifest_path = output_root / "receipt_manifest.json"
    if manifest_path.exists():
        observed = json.loads(manifest_path.read_text())
        # Resume changes only the receipt state counts; identity and SHA rows
        # remain exact.  Replace is forbidden, so report the live validation.
        if (observed.get("dag_payload_sha256")
                != receipts.get("dag_payload_sha256")
                or observed.get("receipts") != receipts.get("receipts")):
            raise SystemExit("existing receipt manifest differs from frozen plan")
    else:
        atomic_json(manifest_path, receipts)
    print(json.dumps({
        "status": "PLANNED_DISABLED_SYNTHETIC_ONLY",
        "dag": str(dag_path), "dag_sha256": sha256_file(dag_path),
        "receipt_count": receipts["receipt_count"],
        "receipt_states": receipts["states"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
