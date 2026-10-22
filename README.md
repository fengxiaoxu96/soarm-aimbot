# soarm-aimbot

An overhead Intel RealSense RGB-D camera tracks an object on the table, and an SO-101 arm pans its base to follow it. A wrist-mounted RGB camera shows the gripper's aim line and where it hits the object or the table.

This implementation is based on the paper [AimBot: A Simple Auxiliary Visual Cue to Enhance Spatial Awareness of Visuomotor Policies](https://arxiv.org/abs/2508.08113) (Dai et al., CoRL 2025). This covers only the hardware setup, calibration and the aim-line visualization on the wrist view. It does not include training or running a VLA policy.

Five calibration steps link the two cameras, the table and the arm. The final command, `aimbot-wrist`, uses their results.

## Hardware

- SO-101 arm with Feetech STS3215 servos on a USB serial bus (e.g. `/dev/ttyACM0`)
- Intel RealSense D455/D455C, fixed above the table and looking down (the overhead camera)
- A USB (V4L2) RGB camera mounted on the wrist, without depth
- Printed targets: a checkerboard, and `DICT_APRILTAG_36h11` AprilTags with ids 0, 2 and 3. The steps can generate the tags for you

## Setup

The project uses [uv](https://docs.astral.sh/uv/) with Python 3.10. `pyproject.toml` lists the dependencies (NumPy, OpenCV, PyYAML, `pyrealsense2`, `pyserial`), and `uv.lock` pins the exact versions. The whole project shares one environment.

1. Install uv if it is not installed yet:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create the environment from the lockfile, with the development extras (pytest):

   ```bash
   cd soarm-aimbot
   uv sync --extra dev
   ```

   This creates `.venv/`, installs Python 3.10 if needed (see `.python-version`), and installs the package in editable mode with all `aimbot-*` commands.

3. Activate it in every new terminal:

   ```bash
   source .venv/bin/activate
   ```

   You can also skip activation and prefix commands with `uv run`, e.g. `uv run aimbot-step01 devices`.

4. Check the installation. No hardware is needed for this:

   ```bash
   pytest
   aimbot-step01 self-check
   ```

To add a dependency, add it to `pyproject.toml` and run `uv sync --extra dev` again.

### System setup for the hardware

- **RealSense:** `pyrealsense2` comes from PyPI, so you do not need to build librealsense. If `aimbot-step01 devices` cannot see a camera that is plugged in, install the udev rules and replug the camera:

  ```bash
  sudo apt install librealsense2-udev-rules
  ```

- **Wrist camera:** OpenCV opens it through V4L2. Your user must be in the `video` group.
- **Servo bus:** your user needs access to the serial port (usually the `dialout` group).
- **Optional:** install `ffmpeg` to re-encode the recorded videos to H.264.

All commands run from the repository root. They read and write `calib/` and `output/` relative to it.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/soarm_aimbot/geometry/` | Hardware-free math: intrinsics, rigid transforms, AprilTag/checkerboard detection, segmentation, tracking, URDF forward kinematics, hand-eye solvers |
| `src/soarm_aimbot/cameras/` | RealSense and V4L2 camera adapters |
| `src/soarm_aimbot/robots/` | Feetech STS3215 position packets |
| `src/soarm_aimbot/steps/` | One runnable command per file. Steps do not import each other |
| `assets/so101/` | Official SO-101 URDF (`so101_new_calib.urdf`). Joint zero is the middle of the travel |
| `calib/` | Calibration results written by the steps and read by later ones |
| `docs/` | Detailed instructions for each step |
| `tests/` | Pytest suite. It runs without cameras or servos |
| `output/` | Images, samples, videos and logs from each run (git-ignored) |

## Coordinate frames

\(T_{a \leftarrow b}\), stored as `T_a_b`, maps points in frame b to frame a.

| Frame | Meaning |
| --- | --- |
| `overhead` / `wrist` | Camera optical center: X right, Y down, Z forward along the optical axis (OpenCV convention) |
| `world` | Table frame. Origin at the center of AprilTag id 0, Z up, X/Y along the arrows printed on the tag |
| `base` | URDF `base_link`. Inverse kinematics and servo commands use this frame |
| `gripper` | URDF `gripper_link`. Step 5 finds the wrist camera's pose relative to it |

## Execution order

Run the steps in this order. Each step reads the results of the steps before it. Every command has `-h` help and a `self-check` subcommand that runs without hardware.

| Order | Command | What it does | Output | Instructions |
| --- | --- | --- | --- | --- |
| 1 | `aimbot-step01` | Aligns overhead depth to color and saves and verifies the overhead intrinsics | `calib/overhead_intrinsics.yaml` | [docs/step01-overhead-intrinsics.md](docs/step01-overhead-intrinsics.md) |
| 2 | `aimbot-step02` | Calibrates the wrist camera intrinsics and distortion with a checkerboard | `calib/wrist_intrinsics.yaml` | [docs/step02-wrist-intrinsics.md](docs/step02-wrist-intrinsics.md) |
| 3 | `aimbot-step03` | Calibrates the overhead camera to the table (world) frame with AprilTag id 0 | `calib/T_world_overhead.yaml` | [docs/step03-table-frame.md](docs/step03-table-frame.md) |
| 4 | `aimbot-step04` | Zeros the servos, then calibrates the overhead camera to the arm base (eye-to-hand) with a temporary AprilTag id 2 on the gripper | `calib/servo_map.yaml`, `calib/T_base_world.yaml` | [docs/step04-base-frame.md](docs/step04-base-frame.md) |
| 5 | `aimbot-step05` | Calibrates the wrist camera to the gripper (eye-in-hand) with AprilTag id 3 fixed on the table | `calib/T_gripper_wrist.yaml` | [docs/step05-wrist-handeye.md](docs/step05-wrist-handeye.md) |
| 6 | `aimbot-wrist` | Tracks the object, pans the base to follow it, and draws the aim line on the wrist view | `output/aimbot/<timestamp>/` | [docs/aimbot-wrist.md](docs/aimbot-wrist.md) |

Notes on the order:

- Steps 1 and 2 are independent of each other. Steps 3–5 depend on the steps before them.
- Step 4 needs a temporary tag on the gripper. Remove it before Step 5.
- If a step fails its own pass criteria, fix that step and run it again. Do not add constant offsets in later steps to make up for the error.
- `calib/` in this repository holds the results from the authors' setup, as an example of the file formats. Run all five steps again on your own hardware. Each step overwrites its own files.

A typical session after calibration:

```bash
source .venv/bin/activate
aimbot-wrist self-check
aimbot-wrist run --preview --port /dev/ttyACM0      # look only, torque stays off
aimbot-wrist run --port /dev/ttyACM0 \
  --color-lower 0,120,60 --color-upper 15,255,255   # drive the arm; press s to start tracking
```

For the details of each step (what to print, how to place targets, keys, pass criteria and troubleshooting), see the documents in `docs/` linked above.

Write arguments that start with a minus sign with `=`, e.g. `--init-pose-deg=-10,0,0,50,-90`.

## Tests

```bash
pytest
```

The tests use synthetic data and the URDF. They do not open cameras or the servo bus.

## License

Apache License 2.0. See [LICENSE](LICENSE).
