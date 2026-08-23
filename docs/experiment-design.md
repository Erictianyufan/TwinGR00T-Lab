# TwinGR00T Lab experiment design v0.1

## Objective

Determine whether GR00T N1.7 post-trained on RoboTwin demonstrations can improve closed-loop success on held-out `stack_bowls_three` initial states and language prompts in Isaac Lab.

This project separates three claims:

1. **Engineering reproduction**: data conversion, training, checkpoint restore, policy serving, and closed-loop evaluation all execute and emit auditable artifacts.
2. **Learning reproduction**: the post-trained checkpoint improves paired held-out outcomes over the unadapted base model.
3. **Research comparison**: GR00T is compared with ACT and pi0.5 under an identical split and evaluation protocol.

Only the first claim is required for the initial milestone.

## Version tracks

### Track A: strict reproduction

- RoboTwin `IsaacLab-Arena` commit from `configs/versions.env`
- Isaac Sim 5.1.0
- Isaac Lab 2.3.x
- Python 3.11

### Track B: migration experiment

- Isaac Sim 6.0.1
- Isaac Lab 3.x

Track B starts only after Track A produces a working closed-loop baseline. Results from the two tracks must not be pooled.

## Task and split

The first task is `stack_bowls_three` with the Aloha embodiment because it is the only task in the official IsaacLab-Arena branch with a public set of 100 converted demonstrations.

| Split | Demo IDs | Purpose |
|---|---:|---|
| Train | 0-79 | post-training |
| Validation | 80-89 | checkpoint selection and debugging |
| Test | 90-99 | final frozen evaluation |

Frames are never randomly split across episodes.

## Canonical modality contract

### Video

- `head_camera_rgb`
- `left_camera_rgb`
- `right_camera_rgb`

The front camera remains available for diagnostics but is excluded from the first policy input.

The converted dataset uses 31.25 Hz: one Isaac environment step is 16 ms and the
official replay recorder stores every second step. RoboTwin's helper video export
computes FPS from the 8 ms render interval instead and therefore reports roughly
62 Hz; TwinGR00T Lab regenerates video from HDF5 at the control-aligned rate so
video, state, action and timestamps remain synchronized.

### State and action

Canonical state and action vectors contain 14 logical dimensions:

```text
[left_arm_6, left_gripper_1, right_arm_6, right_gripper_1]
```

Actual measured joint positions are used as state. The recorder's previous action must not be presented as measured proprioception.

GR00T action semantics:

- arm groups: relative, non-EEF
- gripper groups: absolute, non-EEF
- prediction horizon: 16 steps (a deliberate custom-embodiment override of the N1.7 model default of 40)
- execution horizon: 8 steps

The official N1.7 custom-embodiment example explicitly supports a 16-step
relative joint-space configuration. We use that smaller first-stage horizon to
match the replay controller and reduce closed-loop latency; the 40-step default
is retained as a later ablation, not silently mixed into the baseline.

At the simulator boundary, each one-dimensional gripper command is duplicated to the two physical gripper joints, producing the 16-dimensional Isaac action.

### Language

Training uses `seen` paraphrases. Final language evaluation reports `seen` and `unseen` prompts separately. Prompt selection is fixed in a versioned manifest before evaluating a frozen checkpoint.

## Compute roles

| Resource | Role |
|---|---|
| GPU 0, RTX 4090 24 GB | Isaac Sim, cameras, environment stepping |
| GPU 1, RTX 4090 24 GB | GR00T inference server |
| Separate 40 GB+ GPU job | reference GR00T post-training |

Two 24 GB cards are not treated as one 48 GB card for the reference post-training recipe.

## Acceptance gates

### G0: host and environment

- exact versions and commits recorded in `results/manifests/host.json`
- both GPUs visible and assigned explicitly
- simulator can run 500 zero-action steps headlessly with cameras enabled

### G1: demonstration replay

- five-demo smoke replay completes
- at least four of five smoke demonstrations satisfy the environment success termination
- all 100 public demonstrations are subsequently audited and replay failures are listed

### G2: dataset conversion

- GR00T-flavored LeRobot v2 metadata exists
- state/action shape is 14 for every frame
- timestamps, frame counts, videos, and episode boundaries are consistent
- there are no NaN or Inf values
- relative statistics are generated after the modality configuration is frozen

### G3: policy smoke

- base checkpoint and gated backbone can be loaded
- one real converted batch completes forward inference
- a 2-10 step bounded training run saves and restores a checkpoint

### G4: post-training

- training uses demos 0-79 only
- validation metrics use demos 80-89 only
- exact command, config, commit, GPU, duration, and peak VRAM are recorded

### G5: closed-loop evaluation

- final checkpoint is frozen before prompt manifest generation
- 10 test initial states x 5 fixed prompts = 50 episodes per condition
- success, stage completion, timeout, collision, latency, and action smoothness are reported
- paired differences and 95% confidence intervals are reported

## Initial comparison matrix

| Condition | Required in milestone 1 |
|---|---|
| Unadapted GR00T N1.7 | yes, negative control |
| Post-trained GR00T N1.7 | yes |
| ACT | later |
| pi0.5 | later |
| HIL/online RL | after supervised baseline |

## Known risks

- The official RoboTwin branch uses demo-specific initial-state teleportation and has no randomized reset events for this task.
- The public dataset records a replay-oriented joint action interface, not a ready-made GR00T modality configuration.
- Physics replay can be nondeterministic for contact-rich stacking.
- The selected dual 4090 host is appropriate for simulation plus inference, but not the official single-device post-training memory recommendation.
- Isaac Sim 6 migration is a separate engineering experiment, not a drop-in version upgrade.
