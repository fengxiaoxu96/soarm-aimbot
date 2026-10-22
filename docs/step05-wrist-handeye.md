# Step 5: Wrist Camera to Gripper (Eye-in-Hand)

The wrist RGB camera is mounted on the gripper and moves with the arm. This step looks at a fixed AprilTag on the table from many arm poses and combines what it sees with URDF forward kinematics to find the camera's fixed mount on `gripper_link`. The result goes to `calib/T_gripper_wrist.yaml`, which `aimbot-wrist` uses.

Run all commands from the repository root with the virtual environment activated (see the README).

## What you need

- `calib/wrist_intrinsics.yaml` from Step 2. Keep the same resolution, lens model, and focus as during Step 2. If the file has a `focus` value, it is applied automatically.
- `calib/servo_map.yaml` from Step 4.1 (`servo-zero`). Capture uses it to read the five arm joints from the servo bus.
- `assets/so101/so101_new_calib.urdf`, with the joint zeros already matching it (as in Step 4).
- A printed AprilTag **id 3** (dictionary `DICT_APRILTAG_36h11`), generated below and fixed on the table. Do not use id 0 (the table world origin from Step 3) or id 2 (the Step 4 gripper tag); the command rejects them.
- **Remove the Step 4 gripper tag (id 2)** before you start. Leave the id 0 world marker where it is; do not let the two papers overlap.

## Steps

### 1. Offline checks (no camera, no arm)

```bash
aimbot-step05 self-check
aimbot-step05 joints
aimbot-step05 capture --check
```

`self-check` should print `Self-check passed`, with the gripper origin at all-zero joints at about `(293.2, -0.2, 234.4)` mm in `base_link`. `capture --check` confirms that the capture loop keeps going when the tag is not visible.

### 2. Print and place the table tag

```bash
aimbot-step05 print-marker --size-mm 40
```

This writes `output/step05/wrist_marker.png`. Print it at 100% scale and measure the **outer edge of the black square** with calipers (no white border); use that value as `--size-mm` from now on (e.g. `40.2`). The red arrow is tag +X, the green arrow is +Y.

Fix the paper on the table where the wrist camera can see the whole black square from many directions. It may lie flat or slightly tilted, but it **must not move** during the whole calibration. You do not need to measure where it is; the solver eliminates its position.

### 3. Capture poses by hand

Without `--move`, `capture` only checks the servo map and prints the instructions:

```bash
aimbot-step05 capture --servo-map calib/servo_map.yaml --size-mm 40.2
```

Start the real capture. Use `--reset` the first time to clear an old `output/step05/samples.yaml`:

```bash
aimbot-step05 capture \
  --move --port /dev/ttyACM0 \
  --servo-map calib/servo_map.yaml \
  --size-mm 40.2 \
  --reset
```

Torque stays off on the five arm joints; you move the arm by hand. In the `step05 capture` window and on the terminal line you see the current joint angles, the tag reprojection error, and the number of saved poses.

1. Support the arm and move it until the whole table tag is in the wrist image.
2. When the window shows `tag ready` (reprojection at most 1.5 px), press **s** (or Space).
3. The arm briefly holds its position with torque on while several frames are averaged, then torque is released again. On success the terminal prints `Poses collected: N`. If there were not enough valid frames or a joint was out of its limits, nothing is saved; just move on.
4. Repeat. Aim for **about 20 poses** with clearly different gripper rotations (not just translations), viewing the tag from different positions and angles. Keep the joints away from their limits.
5. Press **q** (or Esc) to finish. Torque is turned off, and if at least 4 poses were saved the solve runs immediately (step 4 below). Ctrl-C also keeps the saved poses.

Use `--device` to pick the camera if there are several (same index as Step 2) and `--focus` to override the focus from the intrinsics file. Running `capture` again without `--reset` appends to the existing samples.

### 4. Solve (if you want to re-run it)

```bash
aimbot-step05 solve
```

It reads `output/step05/samples.yaml`, solves with Park's and Andreff's methods, compares them, and by default holds out the last 4 poses (`--holdout`) to check the result. It prints the camera origin in `gripper_link` and which gripper axis the optical axis is closest to.

### 5. Verify at a new pose (optional)

Move the arm to a pose that was not captured, read the five joint angles (radians), and run:

```bash
aimbot-step05 verify --joints 0.1,-0.4,0.5,-0.2,0.6 --size-mm 40.2
```

It predicts where the tag should appear from forward kinematics and the calibration and compares this with the detection. Tolerances are `--tolerance-mm 5` and `--tolerance-px 1`.

To print the wrist camera pose in `base_link` for given joint angles:

```bash
aimbot-step05 pose --joints 0,0,0,0,0
```

## Output

| File | Content |
| --- | --- |
| `calib/T_gripper_wrist.yaml` | `T_gripper_wrist` (4x4, wrist camera frame to `gripper_link`), the chosen solver, holdout errors, and the optical axis direction in the gripper frame. Used by `aimbot-wrist`. |
| `output/step05/samples.yaml` | Per pose: five joint angles, the tag pose in the wrist camera, and the four corner pixels |
| `output/step05/wrist_marker.png` | Printable table tag (id 3) |

After calibration, do not remove or shift the camera on the gripper; if you do, redo this step.

## Pass criteria

- Park and Andreff agree within 10 mm and 3° (`--agree-mm`, `--agree-deg`). If not, **no file is written** and the command exits with code 1.
- On the held-out poses, the tag origin error is at most 5 mm (`--verify-mm`) and the corner reprojection at most 1 px (`--verify-px`). If this fails, `calib/T_gripper_wrist.yaml` **is still written**, but the command exits with code 1. Recapture with larger rotation differences.
- `verify` (if run) stays within 5 mm and 1 px.

## Troubleshooting

- **"Too few useful rotations" or the solvers disagree:** consecutive poses need clear gripper rotations. Add poses that tilt and roll the wrist, not just ones that slide it.
- **Large holdout or `verify` error:** check that the table tag was not bumped, that `--size-mm` is the measured black-square edge, and that the joint zeros still match the URDF. `aimbot-step05 fk --joints 0,0,0,0,0` gives the expected all-zero gripper position; place the arm at its zero and compare with a ruler. A 1° zero error moves the gripper about 5 mm at full reach.
- **`tag ready` never appears:** the whole black square must be in view and in focus. Move the camera closer or reduce glare on the paper.
- **`... has no joints. Finish servo-zero in Step 4.1 first.`:** `calib/servo_map.yaml` is missing or incomplete; redo Step 4.1.
