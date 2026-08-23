# G1 serving-plate data generation

This stage is an experimental bridge, not a demonstration renderer. It tests
whether Arena's annotated brown-box trajectory can be retargeted to the plate
while preserving physics and native success checks.

## Source-data gate

Audit the official annotated source before every generation campaign:

```bash
python scripts/audit_arena_g1_dataset.py \
  datasets/g1_official/arena_g1_loco_manipulation_dataset_annotated.hdf5 \
  --output results/g1/E2_source_dataset_audit.json
```

Expected source: one successful episode, 855 frames, 23-dimensional actions,
the task-facing object key `brown_box`, and all seven locomotion/manipulation
subtask signals.

## Solver lock

Isaac Sim 5.1 pins OSQP 0.6.7, while recent qpsolvers releases require OSQP
1.x. Install the tested compatibility set without allowing pip to upgrade the
Isaac numerical stack:

```bash
python -m pip install --no-deps -r configs/arena_mimic_constraints.txt
```

Then verify that `qpsolvers.available_solvers` contains `osqp` and solve a
small QP before launching Isaac Sim.

## One-trial feasibility gate

```bash
TWINGR00T_SCENE_USD=$PWD/assets/upstream/galileo_locomanip.usd \
TWINGR00T_SCENE_POSE_XYZ=4.42,1.4008,-0.795 \
TWINGR00T_CARGO_USD=$PWD/assets/serving_plate.usda \
TWINGR00T_CARGO_SCALE=1,1,1 \
TWINGR00T_MIMIC_GUARANTEE=0 \
TWINGR00T_MIMIC_KEEP_FAILED=1 \
TWINGR00T_MIMIC_SEED=1 \
python scripts/generate_g1_plate_dataset.py \
  --headless --enable_cameras --mimic \
  --input_file datasets/g1_official/arena_g1_loco_manipulation_dataset_annotated.hdf5 \
  --output_file datasets/g1_plate/plate_mimic_smoke.hdf5 \
  --generation_num_trials 1 --num_envs 1 --device cpu \
  galileo_g1_locomanip_pick_and_place \
  --object brown_box --embodiment g1_wbc_pink
```

The cargo retains the logical name `brown_box` only to preserve annotation
references. Its USD, collision shape, mass and simulated pose are the serving
plate. Arena exports only episodes that satisfy the native proximity success
predicate; no grasp constraints, teleportation or root-pose animation are
allowed.

Do not start model training unless this gate exports at least one successful
episode and replay validation confirms that the plate is physically carried.
Use fixed-attempt mode for feasibility experiments so failures count toward the
requested trial count. If 0/3 initial attempts pass, classify direct retargeting
as a negative result and create plate-specific demonstrations instead. Enable
the upstream success-guarantee mode only for production expansion after this
gate passes.

## Geometry sweep after direct-retarget failure

The direct brown-box-to-plate retarget gate produced 0/3 successes. The first
inward-X hypothesis also produced no lift, so subsequent candidates vary both
the wrist's object-local X offset and height. Run the bounded sweep with camera
recording disabled; this keeps failed datasets small while retaining physics,
robot state, object state and datagen end-effector poses:

```bash
PYTHON_BIN=/home/ubuntu/env_robotwin/bin/python \
bash scripts/run_g1_plate_geometry_sweep.sh
```

The sweep tests six independent seeds, exports each failed episode, runs
`audit_g1_mimic_failure.py`, and stops at the first native simulator success.
Candidates are evaluated using success, maximum plate lift, plate sliding,
destination distance and wrist-object relative position. A successful candidate
must then be repeated with cameras enabled before it is admitted as training
data.
