# Documentation index

TwinGR00T Lab separates stable runbooks from dated experiment checkpoints. Start
with the architecture and experiment protocol, then use the task-specific pages
for rendering and acceptance details.

## Core documentation

| Document | Purpose |
|---|---|
| [`architecture.md`](architecture.md) | Project framework, runtime boundaries, component map, and evidence flow |
| [`experiment-design.md`](experiment-design.md) | Research question, data split, modality contract, and G0-G5 acceptance gates |
| [`deployment.md`](deployment.md) | Validated dual-4090 cloud setup, pinned versions, bootstrap, and secrets policy |
| [`block-stability.md`](block-stability.md) | Post-settle acceptance criteria for block stacking |
| [`parcel-sorting-demo.md`](parcel-sorting-demo.md) | Physical conveyor feed and contact-grasp warehouse demo |
| [`upstream-contributions.md`](upstream-contributions.md) | RoboTwin contribution audit, evidence, and community submission drafts |

## Experimental branches

| Document | Status |
|---|---|
| [`g1_groot_merge.md`](g1_groot_merge.md) | G1 + GR00T + RoboTwin composition; outside the accepted ALOHA showcase |
| [`g1-n16-single-gpu.md`](g1-n16-single-gpu.md) | Single-48GB-GPU G1 GR00T N1.6 live-inference runbook and evidence contract |
| [`g1-experiment-protocol.md`](g1-experiment-protocol.md) | G1 loco-manipulation experiment protocol |
| [`g1-plate-data-generation.md`](g1-plate-data-generation.md) | G1 serving-plate data generation and geometry sweep |
| [`home_service_demo.md`](home_service_demo.md) | Archived continuous home-service presentation design |

## Checkpoints

Dated checkpoint files are immutable progress snapshots rather than current
runbooks. Use the newest checkpoint for historical context, then confirm commands
and claims against the core documentation and source tree.

- [`EXPERIMENT_CHECKPOINT_2026-08-23.md`](EXPERIMENT_CHECKPOINT_2026-08-23.md)
- [`EXPERIMENT_CHECKPOINT_2026-08-21.md`](EXPERIMENT_CHECKPOINT_2026-08-21.md)

## Documentation rules

- State whether a result is **verified**, **implemented**, **experimental**, or
  **planned**.
- Name the controller that produced visible robot actions.
- Link quantitative claims to a manifest, trace, or acceptance procedure.
- Keep exact setup commands in runbooks and keep the root README concise.
- Never include host passwords, tokens, private dataset URLs, or temporary
  authorization codes.
