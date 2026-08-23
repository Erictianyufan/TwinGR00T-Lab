# G1 loco-manipulation experimental protocol

## Scope

This protocol evaluates whether a learned policy can pick, transport, and place an object with Unitree G1 in
Isaac Lab. A rendered video is an audit artifact, not the success criterion. Root-pose animation, object
teleportation after reset, hidden grasp constraints, replayed joint targets, and edited cross-scene handoffs are
forbidden in reported experimental conditions.

## Conditions

| ID | Scene and object | Policy | Purpose |
|---|---|---|---|
| E0 | Official Galileo scene and brown box | Public Arena-tuned GR00T N1.6 | Reproduction baseline |
| E1 | Same scene with serving plate | Unadapted E0 checkpoint | Zero-shot negative control |
| E2 | Same as E1 | Plate post-trained checkpoint | Learning experiment |
| E3 | Shared G1 + ALOHA scene | Frozen E2 + frozen ALOHA policy | Multi-robot systems experiment |

E1 does not advance until E0 runs correctly. E2 requires a versioned G1 plate dataset. E3 does not advance until
both single-robot policies pass their own frozen evaluation.

## Repetition and controls

- Development: seeds 10-14, one episode per seed.
- Frozen evaluation: at least 20 held-out initial states per condition.
- The same initial states, camera calibration, prompt manifest, episode horizon, and simulator build are paired
  across E1 and E2.
- Model checkpoints and prompt manifests are frozen before evaluation.
- Videos are continuous policy-camera and fixed diagnostic-camera recordings. Cuts are not used to determine
  success.

## Stage metrics

1. `reach`: both wrists enter a fixed region around the object.
2. `contact`: bilateral hand-object contacts persist for a minimum duration.
3. `lift`: object height exceeds its reset height by at least 5 cm without teleportation.
4. `transport`: robot-object displacement exceeds the task threshold while contact persists.
5. `place`: object enters the destination tolerance and remains supported after release.

Every episode records terminal status, maximum lift, minimum destination distance, robot displacement, policy
latency, action delta, contact duration, foot slip, and failure reason. The current runner implements all metrics
except contact duration and foot slip; those are required before E2 is reported.

## Acceptance gates

- **E0-R:** at least 4/5 development seeds complete the official brown-box task without assistance.
- **E1-Z:** the plate zero-shot condition is reported even if success is 0%; it is not repaired with constraints.
- **E2-D:** dataset audit passes, train/validation/test episodes are disjoint, and replay success is at least 90%.
- **E2-L:** the frozen plate checkpoint improves paired success over E1 and reports a 95% confidence interval.
- **E3-S:** G1 and ALOHA share one physics stage and communicate only through observable task state; no object swap
  or edited handoff is permitted.

## Hardware allocation

- GPU 0: Isaac Sim, physics, and cameras.
- GPU 1: policy inference.
- GR00T N1.7 post-training: separate 48 GB or 80 GB GPU host. Dual 24 GB cards are not treated as pooled memory.

The cloud host must have authenticated access to both `nvidia/GR00T-N1.7-3B` and its gated
`nvidia/Cosmos-Reason2-2B` backbone before N1.7 training or inference is marked runnable.
