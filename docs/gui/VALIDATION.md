# GUI integration validation, 2026-09-16

Base algorithm: 17ca600. Existing ssh33 environments and weights reused without installation changes.

- 32 unit/regression tests passed: test_live_gui, test_semantic_refinement, test_semantic_runtime.
- JavaScript node --check and Python parsing passed.
- Independent doc witness executed exact documented remote --help command, exit 0.
- Real Orbbec registered RGB-D 120-frame replay through GUI capture, preview, fresh mapping, SAM3, Qwen NF4, and optional refinement:

| Run | Schedule | run-sam3 seconds | Total with capture/preview/refinement | Points | Semantic >0 | Instance >0 |
|---|---|---:|---:|---:|---:|---:|
| scan_20260916_133246_66c057 | serial | 101.915 | 136.089 | 7487 | 6104 | 3679 |
| scan_20260916_133611_f2f9a5 | parallel | 95.702 | 123.386 | 7487 | 6104 | 3679 |

Remote results: `/home/aidenwu/Documents/SGF-developnew-GUI-scans/<run>/pipeline/GUI_RESULT.json`. All final PLYs contain xyz, RGB, semantic_id, instance_id, confidence, naming_support. Coverage is not accuracy; no new GT quality claim. One repeat per schedule; capture cadence and UI revision differ, not a controlled speed benchmark.

Browser verified final 7,487-point rendering and semantic/instance/RGB switches. Start/cancel tested through GUI: scan_20260916_133551_c71ffe preserved 74 captured frames; its capture and preview PIDs exited. No forced cancellation under unresponsive hardware tested. Camera uses inherited hardware D2C path; live physical camera scan remains unverified in this integration.

Second run includes latest class legend and GUI progress reporting. Final HTML-only legend visibility adjustment checked by reload. Local API binding remains loopback with per-session token for mutations. Old GUI directory left intact.
