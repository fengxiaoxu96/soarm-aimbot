# Aimbot: Track an Object and Aim the Wrist Camera

`aimbot-wrist` is the final command. The overhead camera segments and tracks an object on the table, the arm pans its base to follow it, and the wrist-camera view shows where the gripper is aiming. It reads the calibration from Steps 1–5 and writes videos and a log to `output/aimbot/`.

Run all commands from the repository root with the virtual environment activated (see the README).

## What you need

- Steps 1–5 finished. Every one of these files must exist:

  | File | Written by |
  | --- | --- |
  | `calib/overhead_intrinsics.yaml` | Step 1 |
  | `calib/wrist_intrinsics.yaml` | Step 2 |
  | `calib/T_world_overhead.yaml` | Step 3 |
  | `calib/T_base_world.yaml`, `calib/servo_map.yaml` | Step 4 |
  | `calib/T_gripper_wrist.yaml` | Step 5 |

- The overhead RealSense, the wrist camera and the SO-101 servo bus connected. The world AprilTag (id 0) can stay on the table. Remove the Step 4 gripper tag and the Step 5 board.
- A target object between 5 and 300 mm tall (`--min-height-mm`, `--max-height-mm`). A brightly colored object is recommended so you can use a color filter.
- A graphical display. Without one, pass `--frames N --no-window` and the run stops after N frames.

## How it works

1. **Stage 1 – initial pose.** All five joints ramp to a fixed pose (default `0,0,0,50,-90` degrees for pan, lift, elbow, wrist_flex, wrist_roll) and hold it under torque. Detection already runs, so you can place the object while you watch the overlay.
2. **Stage 2 – pan trace.** Press `s`. Joints 2–5 stay locked and only `shoulder_pan` turns, so the gripper's base-frame y follows the object's base-frame y.
3. **Wrist overlay.** A ray is cast from the gripper along its −Z axis (the approach direction). The first hit, either a sphere around the tracked object (`--object-radius-mm`) or the table plane, is the aim point. It is drawn as a dot with a green crosshair. The line is blue when it hits the object, red when it hits the table, and orange when it hits nothing within 0.6 m.

## Steps

### 1. Self-check (no hardware)

```bash
aimbot-wrist self-check
```

Should print `Self-check passed`. This only checks the solver and ray geometry against the URDF.

### 2. Preview without moving the arm

```bash
aimbot-wrist run --preview --port /dev/ttyACM0 \
  --color-lower 0,120,60 --color-upper 15,255,255
```

`--preview` never enables torque or writes the servos. With `--port`, the overlay uses the real joint angles. Without `--port`, it uses example joint angles, so the wrist overlay will not match the image. Check that:

- the `overhead` window draws a box with `id<N>` around the object, and its world coordinates change when you move it;
- the `wrist aimbot` window shows the `target` circle on the object.

Color thresholds are OpenCV HSV (H 0–179, S and V 0–255). The example selects red. Without `--color-lower`/`--color-upper`, every object above the table is a candidate. The arm is then masked out of the depth image with `--arm-radius-mm`. Use `--xy xmin,ymin,xmax,ymax` (world metres) or `--ignore-roi x,y,w,h` (overhead pixels, repeatable) to exclude clutter.

### 3. Run on the arm

```bash
aimbot-wrist run --port /dev/ttyACM0 \
  --color-lower 0,120,60 --color-upper 15,255,255
```

Torque turns on at the current position and the arm moves to the initial pose at `--init-speed-deg-s` (default 30). When the terminal prints `Initial pose reached`, place the object and press `s`. Move the object left and right: the base pans to follow it and the aim line moves on the wrist view.

| Key | Action |
| --- | --- |
| `s` | Stage 1 → Stage 2 (only after the initial pose is reached); Stage 2 → back to Stage 1 |
| `r` | Release torque (the arm can be moved by hand) |
| `e` | Re-enable torque at the current position |
| `q` / `Esc` | Quit. Torque is released on exit |

Useful options:

- `--init-pose-deg=0,0,0,50,-90`: another initial pose (URDF degrees, must be within the joint limits).
- `--speed-scale 0.5`: slower pan in Stage 2 (fraction of the URDF velocity limit, default 0.8).
- `--track-id N`: follow a specific track instead of the first confirmed one.
- `--device N`, `--serial S`: select the wrist camera (`/dev/videoN`) or the overhead RealSense when several are connected.

## Output

Each run writes a timestamped folder `output/aimbot/YYYYmmdd-HHMMSS/`:

| File | Contents |
| --- | --- |
| `overhead.mp4` | Overhead view with detections and the HUD |
| `wrist.mp4` | Wrist view with the aim line |
| `log.jsonl` | One JSON line per frame: mode, reason, track, object position in world and base frames, gripper pose, joint angles |

Videos are re-encoded to H.264 when `ffmpeg` is installed. Otherwise the original file is kept.

## Troubleshooting

- **`driving the arm needs --port`**: give `--port`, or add `--preview`.
- **`no display found`**: add `--frames N` (and `--no-window`).
- **HUD shows `no confirmed track`**: the object is not detected for 3 frames in a row. Check the color thresholds, the height band, and that the object is inside `--xy`.
- **HUD shows `target base y ... is outside the reachable range`**: with the locked joints the gripper cannot reach that y. Move the object closer or choose another `--init-pose-deg`.
- **The aim line or target circle is offset from the image**: recheck Step 5 (hand-eye) and that `--port` is given so the real joint angles are used.
- **`s` is ignored in Stage 1**: wait until the arm stops at the initial pose. If torque was released with `r`, press `e` first.
