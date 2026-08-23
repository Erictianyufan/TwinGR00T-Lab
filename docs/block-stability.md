# Block-stack stability acceptance

RoboTwin's standard stack success term is primarily positional. A replay can
therefore be marked successful while the top object is tilted, still moving, or
being supported by a gripper. This project adds a post-settle acceptance gate so
the final showcase does not contain a stack that is already falling.

## Acceptance procedure

1. Replay the complete recorded action sequence in Isaac Lab.
2. Hold the final action for 30 additional control steps.
3. Sort the blocks by world-frame height.
4. Check the XY displacement between every adjacent pair.
5. Check each block's local Z axis against the world Z axis.
6. Check maximum linear and angular velocity after settling.
7. Export the demonstration only when both the task success term and all
   stability checks pass.

The current presentation thresholds are:

- adjacent-block XY offset: at most 15 mm;
- minimum upright cosine: at least 0.994;
- maximum linear speed: at most 0.02 m/s;
- maximum angular speed: at most 0.20 rad/s.

The renderer repeats the validation at the end of the recorded video and writes
the measured values to the adjacent JSON manifest. A failed validation produces
a failed manifest and the video is not promoted to the canonical showcase.

## Reproduction

The cloud deployment copies `remote_overrides/record.py` into the upstream
RoboTwin checkout before conversion. The five-camera renderer uses
`scripts/replay_live_multiview.py --validate-block-stability` and includes a
post-roll long enough to show the released stack at rest.
