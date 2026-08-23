# TwinGR00T HomeRelay demo

`home_service_demo.py` builds the first continuous multi-robot collaboration demo on top of the verified
`stack_bowls_three` IsaacLab-Arena environment.

## Version 1 scope

The first version deliberately separates cinematic integration risk from policy-training risk:

1. A single Isaac Sim stage contains an open-plan kitchen, living room, dining table, three seated family members,
   the ALOHA workcell, and a dependency-free humanoid proxy.
2. The humanoid proxy transports a tray of three visual bowls from the kitchen to the handoff zone.
3. At the handoff boundary, visual cargo and the recorded physical bowls occupy the same poses. The visual cargo is
   hidden and the physical bowls are enabled without a visible teleport.
4. The verified 600-step ALOHA demonstration stacks all three physical bowls.
5. The same simulator, physics timeline, environment, and video encoder remain active for the full sequence.

The proxy isolates scene/orchestration work from the G1 dependency. Version 2 replaces only the
`/World/HomeRelay/HumanoidCarrier` root and its deterministic transform with a Unitree G1 USD and its controller.

Version 2 is now implemented. See [g1_groot_merge.md](g1_groot_merge.md) for the accepted real-G1 render, policy
routing, Isaac People assets, and the remaining live-inference authentication gate.

## Run on the cloud workstation

From the project root:

```bash
source /home/ubuntu/env_robotwin/bin/activate
python scripts/home_service_demo.py \
  vendor/RoboTwin/data/stack_bowls_three/data/demo_0.hdf5 \
  --output artifacts/videos/home_relay_demo_v1.mp4 \
  --headless --device cuda:0 --rendering_mode quality \
  --environment manip_eval_tasks.examples.manipulation.stack_bowls_three_environment:StackBowlsThreeEnvironment \
  stack_bowls_three --embodiment aloha --enable_cameras True
```

For a quick scene check, add `--max-frames 10` and use a different output filename. The complete demo contains
1,000 frames at 31.25 FPS, or 32 seconds.

## Acceptance criteria

- The overview camera shows the kitchen, dining family, carrier, ALOHA station, and handoff route.
- No simulator restart, scene cut, or object teleport is visible in the output.
- The carrier and tray reach the handoff zone before ALOHA moves.
- The three physical bowls appear at the recorded initial positions on the handoff frame.
- The ALOHA replay ends with all three bowls stacked.
- The manifest reports all five phases and all five cameras.

## Version 2 policy replacement

The continuous timeline and handoff contract remain unchanged. Replace the deterministic proxy with:

- a Unitree G1 articulation and GEAR-SONIC whole-body controller;
- a G1-specific GR00T modality configuration and policy server;
- a tray pose condition that advances the coordinator from `humanoid_transport` to `handoff`;
- recovery states for dropped cargo, navigation timeout, and blocked handoff zones.

ALOHA remains a separate GR00T embodiment and policy contract. The coordinator dispatches language instructions and
observes completion predicates; it does not merge the two robots' action spaces.
