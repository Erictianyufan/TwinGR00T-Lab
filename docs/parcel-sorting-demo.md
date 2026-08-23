# Isaac warehouse parcel-sorting demo

This presentation demo places the ALOHA bimanual workcell inside the Isaac
Digital Twin Warehouse and references an official Isaac conveyor asset. Three
visible parcel models travel along the belt, enter the two-arm workspace, and
are delivered to priority and standard bins. The third parcel includes a
left-to-right cooperative handoff.

The first implementation intentionally separates visual validation from policy
claims:

- the robot uses a successful recorded RoboTwin bimanual action sequence;
- recorded actions are continuously resampled and temporally filtered instead
  of snapped to integer source frames;
- parcel paths are deterministic, continuous showcase choreography attached to
  a cached two-pass end-effector trace;
- inter-arm handoff follows a smooth raised arc and bin placement follows a
  lift-transfer-lower clearance path;
- package decoration uses parent-local transforms, preventing child geometry
  from receiving the package world offset twice;
- the referenced conveyor is an Isaac production visual asset, but this version
  does not claim surface-velocity conveyor dynamics;
- the renderer writes these execution modes into the JSON manifest.

This makes the video reproducible and visually inspectable before adding an
operational-space controller, contact-based grasp attachment, conveyor
OmniGraph, and GR00T policy inference.

## Render command

```bash
source /home/ubuntu/env_robotwin/bin/activate
python scripts/parcel_sorting_demo.py \
  vendor/RoboTwin/data/blocks_ranking_rgb/data/demo_0.hdf5 \
  --output artifacts/videos/parcel_sorting_demo_v1.mp4 \
  --headless --device cuda:0 --rendering_mode quality \
  --environment manip_eval_tasks.examples.manipulation.blocks_ranking_rgb_environment:BlocksRankingRgbEnvironment \
  blocks_ranking_rgb --embodiment aloha --enable_cameras True
```

The expected output is a 14-second, five-camera H.264 video plus an adjacent
JSON manifest naming the exact warehouse and conveyor USD paths. The manifest
also rejects runs with a parcel translation above 0.05 m per output frame or a
parcel height below 0.70 m, and records the exact worst frame for review.
