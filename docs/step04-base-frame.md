# Step 4: Overhead Camera to Arm Base

This step calibrates the fixed overhead camera to the SO-101 arm's `base_link` (eye-to-hand). A temporary AprilTag on the gripper is seen by the overhead camera while you move the arm by hand; the joint angles come from the servos and URDF forward kinematics. It writes `calib/servo_map.yaml` (servo zero ticks and directions) and `calib/T_base_world.yaml` (`T_base_overhead` and `T_base_world`, which maps table coordinates into `base_link`).

Run all commands from the repository root with the virtual environment activated (see the README).

## What you need

- `calib/overhead_intrinsics.yaml` (Step 1) and `calib/T_world_overhead.yaml` (Step 3). The id 0 world marker stays on the table, and the overhead camera must not move.
- An SO-101 arm with Feetech STS3215 servos on a serial port (e.g. `/dev/ttyACM0`). The five arm joints default to ids `1,2,3,4,5` in the order `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll` (`--ids`); the gripper servo is not used. Baud rate defaults to 1000000.
- The URDF `assets/so101/so101_new_calib.urdf` (the default). Its q=0 is the calibrated zero pose, not a mechanical stop.
- A printed gripper AprilTag, id 2 (generated below), and calipers. Never use the id 0 table marker on the gripper.
- A graphical display for the capture preview.

## Steps

### 1. Offline checks (no camera, no servos)

```bash
aimbot-step04 self-check
aimbot-step04 joints
aimbot-step04 servo-zero --check
aimbot-step04 servo-capture --check
```

All four should pass. `self-check` reports the all-zero gripper position, about `(293.2, -0.2, 234.4)` mm.

### 2. Print the gripper marker

```bash
aimbot-step04 print-marker --size-mm 40
```

This writes `output/step04/gripper_marker.png`. Print at 100%, then measure the outer edge of the black square with calipers (white border excluded) and use that value as `--size-mm` below. The red arrow is marker +X, the green arrow +Y.

### 3. Mount the marker on the gripper

Stick it flat on a rigid part that moves with the wrist, facing up toward the overhead camera so the whole black square stays visible:

- **On the gripper body (`gripper_link`, the default tip link):** use the fixed jaw / palm, never the moving jaw (gripper opening is not part of the kinematic chain).
- **On the wrist housing (`wrist_link`):** if the palm is too small, this works too, but then add `--tip-link wrist_link` to every capture command. A tag on the wrist housing does not turn with `wrist_roll`.

Measure the mounting pose relative to the tip link: `--tag-xyz-mm` is the marker center in mm along the link's axes, and `--tag-rpy-deg` is its roll,pitch,yaw (0,0,0 when red arrow = link +X, green = link +Y, and the printed face = link +Z). Print the link axes at the zero pose with:

```bash
aimbot-step04 fk --joints 0,0,0,0,0
aimbot-step04 fk --tip-link wrist_link --joints 0,0,0,0,0
```

At q=0, `gripper_link` has its origin on the wrist-roll axis, +X pointing up, +Y along base +Y, and +Z along base −X. The fingertip is about 98 mm further along −Z. The solve itself does not need the mounting pose. It is used for the held-out pose check, so an error here shows up as a large held-out error.

### 4. Step 4.1: zero the servos (`servo-zero`)

Keep room around the arm and be ready to support it by hand.

```bash
aimbot-step04 servo-zero --move --port /dev/ttyACM0
```

Torque is switched off on the five arm joints. The joints are then done one by one in order:

1. The terminal prints the joint's positive direction (as fingertip motion in `base_link`) and a live tick readout on one line.
2. Move the joint to q=0 by hand and press Enter (the tick is read five times and the median is kept):
   - `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`: push against one stop and note the tick, push against the other stop and note it, then set the joint to the average of the two.
   - `wrist_roll`: note both stop ticks as well. The negative stop is the one opposite the printed positive motion. q=0 lies 49.1% of the way from the negative stop to the positive stop, about 2.8° from the midpoint toward the negative side. Compare the gripper orientation with `fk --joints 0,0,0,0,0`.
3. Turn **only that joint** about 10° in the printed positive direction and press Enter. This records the sign.
4. The joint drives itself back to its zero tick and holds there with torque on. Support the arm; the remaining joints are still free.

When all five are done, `calib/servo_map.yaml` is written and torque is switched off. Zero accuracy matters: 1° of zero error gives about 5 mm at the gripper. If one joint is wrong, re-run the command; it rewrites the whole file, so all five joints are redone.

Optional check: drive the arm to q=0 and compare it with the `fk` description above (`--release` turns torque off once it arrives):

```bash
aimbot-step04 move-zero --move --port /dev/ttyACM0 --release
```

### 5. Step 4.2: capture poses and solve (`servo-capture`)

For the first run, add `--reset` to clear the old `output/step04/samples.yaml`. Use your measured values; drop `--tip-link wrist_link` if the tag is on the gripper body.

```bash
aimbot-step04 servo-capture \
  --move --port /dev/ttyACM0 \
  --size-mm 40 \
  --tag-xyz-mm 0,-20,10 \
  --tag-rpy-deg 0,0,0 \
  --tip-link wrist_link \
  --reset
```

Torque stays off and you move the arm by hand. The `step04 servo-capture` window and the terminal show the joint angles, the reprojection error, and the number of saved poses.

1. Pose the arm so the tag is fully visible. Vary shoulder, elbow and wrist so that consecutive poses differ by a clear rotation, not just a translation. Keep away from the joint limits.
2. When the preview shows `tag ready` (reprojection ≤ 2 px), press **s** (or Space). The arm briefly holds with torque while the marker pose is averaged over several frames, then torque is released again. If too few frames are valid, nothing is saved; just keep posing.
3. Collect 15–20 poses. The last 3 are held out for verification (`--holdout`), so you need at least 7.
4. Press **q** (or Esc) to finish. Torque is switched off and the saved poses are solved immediately, writing `calib/T_base_world.yaml`. With fewer than 4 poses it only saves. Ctrl-C also keeps the saved poses.

To solve again from the saved samples (e.g. with a different `--holdout`):

```bash
aimbot-step04 solve
```

### 6. Sanity check and clean up

Map the table origin (or any measured table point) into `base_link` and compare with a tape measure from the arm base:

```bash
aimbot-step04 map --xyz-mm 0,0,0
```

Then **remove the gripper marker** (or make sure it does not block the wrist camera). Leave the id 0 table marker in place.

## Output

| File | Content |
| --- | --- |
| `calib/servo_map.yaml` | Per joint: servo id, `tick_at_urdf_zero`, `sign`. Used by Step 5 and `aimbot-wrist` |
| `calib/T_base_world.yaml` | `T_base_overhead`, `T_base_world`, solver agreement and held-out errors. Used by `aimbot-wrist` |
| `output/step04/samples.yaml` | Captured joint angles and marker poses (input to `solve`) |
| `output/step04/gripper_marker.png` | Printable gripper marker |

## Pass criteria

- Park and Andreff solutions agree within 20 mm and 5° (`--agree-mm`, `--agree-deg`). Otherwise nothing is written.
- Held-out marker translation error is at most 15 mm (`--verify-mm`). Otherwise the file is written but the command exits with code 1.

## Troubleshooting

- **The two solvers disagree:** add poses with more rotation between them (shoulder, elbow and wrist all moving).
- **Held-out error above 15 mm:** check `--tag-xyz-mm` / `--tag-rpy-deg`, that `--tip-link` matches where the tag is mounted, and the joint zeros (redo Step 4.1). Then recapture with `--reset`.
- **Joints show `out of limits or read failed`:** the servo map is wrong or stale. Check the port and ids, and redo Step 4.1.
- **Tag never `ready`:** check `--size-mm` and the marker id, make sure the tag faces the camera and is not covered by the fingers, and avoid glare.
