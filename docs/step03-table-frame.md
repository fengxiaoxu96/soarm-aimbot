# Step 3: Overhead Camera to Table Frame

This step fixes the table coordinate frame with one AprilTag taped to the table and calibrates the fixed overhead camera to it. It writes `calib/T_world_overhead.yaml`, the 4x4 transform that maps points from the overhead camera frame into the table (world) frame. The arm and the wrist camera are not used.

Run all commands from the repository root with the virtual environment activated (see the README).

## What you need

- The overhead RealSense from Step 1, in the same mount and at the same resolution, and `calib/overhead_intrinsics.yaml` from Step 1. The resolution is read from that file.
- A printer and calipers.
- Two printed AprilTags (generated below, dictionary `DICT_APRILTAG_36h11`):
  - **World marker, id 0.** Taped flat on the table, pattern up, fully visible to the overhead camera with its white border. Its center is the world origin, the printed red arrow is world +X, the green arrow is world +Y, and Z points up toward the camera. It stays on the table permanently: later steps and the start-up check depend on it. Do not move it once calibrated.
  - **Verification marker, id 1.** Only needed temporarily for `verify`; remove it afterwards.
- Optional: a graphical display. `calibrate`, `check`, and `verify` show the detection when one is available, and work without one.

Global options (`--intrinsics`, `--dictionary`, `--marker-id`, `--serial`, `--name-substring`) go **before** the subcommand, e.g. `aimbot-step03 --serial <serial> calibrate --size-mm 118.5`.

## Steps

### 1. Optional math check (no camera)

```bash
aimbot-step03 self-check
```

It should print `Extrinsic check passed`. This only checks the math, not a real calibration.

### 2. Generate and print the markers

```bash
aimbot-step03 print-markers --size-mm 120
```

This writes `output/step03/world_marker.png` (id 0, with +X/+Y arrows) and `output/step03/verify_marker.png` (id 1). `--size-mm` is the design side length of the black square. About 120 mm works well with the camera 0.6–1.0 m above the table; much smaller markers give jittery poses.

- Print at 100% / actual size. Turn off "fit to page" and any scaling.
- Measure the **outer edge of the black square** with calipers, excluding the white border. Use this measured value (e.g. `118.5`) as `--size-mm` from now on. Print both markers in the same job so they share one measured size.
- Keep the markers flat and non-reflective (no glossy lamination). Do not draw on the black square.

### 3. Tape down the world marker

Place the world marker where you want the table origin, pattern up, flat on the table, and well inside the camera image (not at the edge). The origin is the center of the black square, not an arrow tip or a paper corner. Do not touch the camera or the marker during calibration.

### 4. Calibrate

```bash
aimbot-step03 calibrate --size-mm 118.5
```

Replace `118.5` with your caliper value. The command warms up the camera, collects about 30 good detections (`--samples`), averages them, and writes `calib/T_world_overhead.yaml`. Keep everything still. In the preview window, press `q` or Esc to abort without writing the file.

At the end it prints:

- the camera center in the table frame (Z should be positive, roughly the camera height);
- the optical axis in the table frame, which should be close to `(0, 0, -1)` for a downward-looking camera;
- the median reprojection error and translation jitter;
- the median world height of table points around the marker, which should be close to 0 mm (a warning appears above about 15 mm).

### 5. Verify with the second marker

Tape the verification marker flat on the same table surface, not covering the world marker; both must be fully visible. Measure from the **world marker center to the verification marker center**:

- `--x-mm`: distance along the red +X arrow (negative if the marker is on the opposite side);
- `--y-mm`: distance along the green +Y arrow, with the same sign rule;
- `--z-mm`: leave at 0 when both markers lie on the same surface.

```bash
aimbot-step03 verify --x-mm 200 --y-mm 50
```

It prints the estimated and measured positions and the error. The default tolerance is 10 mm (`--tolerance-mm`). Remove the verification marker afterwards.

### 6. Re-check at every start-up

Before using table coordinates in a new session (or after anything was bumped):

```bash
aimbot-step03 check
```

This re-detects the world marker and compares with the saved file. It passes (exit code 0) if the change is within 5 mm and 1° (`--trans-mm`, `--rot-deg`); otherwise it exits with code 1 and leaves the file unchanged. Put the marker and camera back and run it again. If the new position is the one you want to keep, make sure the marker is taped down and the camera is fixed, then overwrite the file:

```bash
aimbot-step03 check --update
```

## Output

| File | Content |
| --- | --- |
| `calib/T_world_overhead.yaml` | Overhead-to-table transform, used by Step 4 and `aimbot-wrist` |
| `output/step03/world_marker.png`, `verify_marker.png` | Printable markers |
| `output/step03/detection.png` | Last calibration frame with the marker axes drawn; check that they match the printed arrows |
| `output/step03/verify.png` | Verification frame: world marker in green, verification marker in yellow |
| `output/step03/detection_failed.png` | Written only when the marker could not be detected |

## Pass criteria

- `verify` error is at most 10 mm.
- `check` reports at most 5 mm translation and 1° rotation.
- The optical axis is close to `(0, 0, -1)` and the camera center has positive Z.
- The table height around the marker is close to 0 mm (informational only; depth is often missing on the black pattern).

## Troubleshooting

- **Marker not detected:** look at `output/step03/detection_failed.png`. Usually the marker is partly outside the image, the white border is cut off, or the surface reflects light. `--size-mm` does not affect detection, only the computed distances.
- **`verify` error too large:** check that `--size-mm` excludes the white border, and that the signs of `--x-mm` / `--y-mm` follow the printed arrows. Then re-run `calibrate`.
- **`check` fails after bumping the camera or marker:** do not keep using the old file. Restore the setup, or re-run `calibrate` (or `check --update`) for the new setup. Step 4 builds `calib/T_base_world.yaml` on top of this file, so redo Step 4 if the table frame or the camera changed.
