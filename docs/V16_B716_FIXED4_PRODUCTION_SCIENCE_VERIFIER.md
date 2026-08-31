# Fixed4 production science verifier

`v16_b716_fixed4_production_adapters.finalize_v15_from_slot_results` does not
trust `raw_summary.cross_solver_check.rotation_deg/translation_m` as an
observation.  Those fields are derived output and are accepted only after the
adapter independently reproduces them from the selected slot's 20 worker JSON
atoms.

The verifier requires the canonical `2 solvers x 2 directions x 5 repeats`
worker filenames.  Every worker must be a non-symlink regular file, have a
valid self-hash, match the raw summary's per-worker evidence hash and
cache/runtime provenance, and satisfy the V13 GT-free/fail-closed contract.
Missing, additional, malformed, non-finite, or modified evidence fails closed.

For each solver and direction the verifier independently reconstructs the
complete-linkage 4-of-5 component and its observed medoid.  It checks the
recorded gate, medoid repeat, and medoid transform.  The public SE(3) convention
is column-vector `reference = R @ source + t`: forward medoids map source to
reference; reverse medoids map reference to source and are inverted only for
the forward/reverse direction check.  The finite cross-solver measurement is
the distance between the recomputed PointDSC-forward and
pyGCRANSAC-forward medoids.

Recorded rotation and translation may differ from recomputation only by
absolute roundoff (`1e-9` degree and `1e-12` metre, relative tolerance zero).
Transform bindings use absolute matrix tolerance `1e-10`.  The selected V15
observed transform must also equal the named strict final realization (with an
explicit inverse for a reverse realization).  Direction drift, summary
tampering, worker tampering, or a substituted V15 medoid therefore cannot steer
the result.

The frozen finite gate is unchanged: a recomputed rotation greater than 5
degrees or translation greater than 0.10 metre is `FAIL`; exact-boundary and
smaller values are `PASS`.  Any absence or semantic inconsistency raises a
typed adapter failure, which the active wrapper converts to a fail-closed
outcome without a transform.
