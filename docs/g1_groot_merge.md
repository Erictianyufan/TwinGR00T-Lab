# G1 + GR00T + RoboTwin merged demo

The version-2 HomeRelay demo is one uninterrupted Isaac Sim stage rather than two edited videos.

## Runtime composition

- **Unitree G1:** Isaac Lab's official `Robots/Unitree/G1/g1.usd` articulation. The renderer writes named hip,
  knee, ankle, shoulder, and elbow targets, so motion remains independent of the USD's internal joint ordering.
- **G1 policy route:** GR00T N1.7 `REAL_G1` (`real_g1_relative_eef_relative_joints`) on port 5556. The current
  cinematic uses the same route contract with a deterministic gait adapter. Replacing the adapter with
  `UNITREE_G1_SONIC` requires a task-specific posttrained checkpoint and the GEAR-SONIC WBC.
- **ALOHA route:** RoboTwin's 16-joint ALOHA environment and `NEW_EMBODIMENT` GR00T contract on port 5555. The
  accepted video replays the verified 600-step `stack_bowls_three` trajectory.
- **Coordinator:** G1 owns pickup, transport, and handoff. ALOHA owns stacking and completion. Visual cargo is
  swapped for the physical RoboTwin bowls at the handoff pose inside the same simulation timeline.
- **People:** three NVIDIA Isaac People/AnimGraph USD assets. Runtime bounds are recorded in the JSON manifest to
  catch mixed centimeter/meter units and incorrect skeleton axes.

## Cloud paths

- Project: `/home/ubuntu/mycode/robotwin-gr00t-lab`
- GR00T checkpoint: `checkpoints/GR00T-N1.7-3B`
- Final video: `artifacts/videos/home_relay_g1_groot_multiview_v2.mp4`
- Manifest: `artifacts/videos/home_relay_g1_groot_multiview_v2.json`
- Render log: `logs/home_relay_g1_groot_multiview_v2.log`

## Reproduce the accepted render

```bash
/home/ubuntu/env_robotwin/bin/python scripts/home_service_demo.py \
  vendor/RoboTwin/data/stack_bowls_three/data/demo_0.hdf5 \
  --output artifacts/videos/home_relay_g1_groot_multiview_v2.mp4 \
  --headless --device cuda:0 --rendering_mode quality \
  --environment manip_eval_tasks.examples.manipulation.stack_bowls_three_environment:StackBowlsThreeEnvironment \
  stack_bowls_three --embodiment aloha --enable_cameras True
```

The application waits for remote USD payloads and textures before encoding. This is required because recording
before the payload queue drains can produce valid geometry bounds but blank frames.

## Live GR00T inference gate

The full N1.7 checkpoint is present, but N1.7 also loads the gated `nvidia/Cosmos-Reason2-2B` backbone. The cloud
machine must be authenticated with a Hugging Face read token whose account has access before the REAL_G1 server can
start. After authentication, launch it on the second 4090:

```bash
CUDA_VISIBLE_DEVICES=1 /home/ubuntu/env_groot/bin/python -m gr00t.eval.run_gr00t_server \
  --model-path checkpoints/GR00T-N1.7-3B \
  --embodiment-tag REAL_G1 --device cuda --host 127.0.0.1 --port 5556
```

Do not select `UNITREE_G1_SONIC` with the base checkpoint: it is a posttrain tag and requires its matching
fine-tuned model plus the GEAR-SONIC whole-body-control stack.
