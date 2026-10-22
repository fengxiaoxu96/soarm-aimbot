# Step 2: Wrist Camera Intrinsics

This step calibrates the intrinsics and lens distortion of the wrist RGB camera with a checkerboard and writes them to `calib/wrist_intrinsics.yaml`. The overhead camera is not used, and no joint angles are recorded; the camera-to-gripper pose comes later, in Step 5.

Run all commands from the repository root with the virtual environment activated (see the README).

## What you need

- The wrist RGB camera mounted on the arm and plugged into USB. It appears as a `/dev/video*` node, mixed with the RealSense nodes.
- Your user in the `video` group (see Troubleshooting).
- A printed checkerboard on a flat, rigid board (the Step 1 board works). Print at 100% scale with at least one square of white margin, on matte paper. Squares of 15–25 mm with about 9x6 inner corners suit a wide-angle wrist camera. Measure one printed square with calipers.
- A graphical display for `preview`, `calibrate`, and `undistort`. `devices`, `probe`, `self-check`, and `show` work without one.

Camera options (`--device`, `--width`, `--height`) go **before** the subcommand, for example `aimbot-step02 --device 4 probe`. The default resolution is 1280x720.

## Steps

### 1. Find the wrist camera

```bash
aimbot-step02 devices
```

Each line is a `/dev/videoN` node. Nodes marked `RealSense node` belong to the overhead camera; the wrist camera is an `RGB candidate`. With only one RGB candidate you can omit `--device`; otherwise pass its number, e.g. `--device 4`.

### 2. Pick a resolution

```bash
aimbot-step02 --device 4 probe
```

This tries 640x480, 800x600, 1280x720, and 1920x1080 and prints the ones the camera actually delivers. Pick one and use it for calibration and for every later wrist image. Changing the resolution after calibration makes the intrinsics invalid.

### 3. Focus

```bash
aimbot-step02 --device 4 --width 1280 --height 720 preview
```

Fix the board on the table, move the arm to the working distance, and press `[` / `]` to change focus in steps of 5 until the squares are sharp. Note the `focus` number shown at the top left and press `q`. If the number does not change, the lens has no software focus: leave the focus ring alone (tape it if needed) and omit `--focus` below.

### 4. Capture and calibrate

`--cols`/`--rows` are **inner corners** (10x7 squares = 9x6 inner corners), and `--square-mm` is the measured printed square size.

```bash
aimbot-step02 --device 4 --width 1280 --height 720 calibrate \
  --cols 9 --rows 6 --square-mm 25 --focus 40
```

Keep the board fixed and move the arm (by hand or joint by joint) to change the camera pose. For each pose, wait until the arm and image are still, check that colored corner lines are drawn, and press `SPACE`. Vary the pose throughout:

- one roughly square-on view with the board filling about half the image;
- views that put the board near each corner and each edge of the image;
- views tilted about 30° about both axes;
- slightly nearer and farther, but always sharp and with the whole board and its margin in view.

At least 20 views are required (`--views`); 4 evenly spaced views are held out to check the result (`--holdout`). Press `c` to compute or `q` to abort without writing a file. If the terminal says all board centers are in the middle of the image, add poses near the edges.

If straight lines bend strongly near the image edges, use the fisheye model by adding `--model fisheye` (the default is `brown`). Use the same model in all later steps.

### 5. Check the result

```bash
aimbot-step02 --device 4 undistort
aimbot-step02 show
```

`undistort` opens the camera at the calibrated resolution and focus and shows raw (left) and undistorted (right) images; straight board edges should look straight on the right. Press `q` to quit. `show` prints the saved intrinsics, the held-out error, and the principal-point ray, which should be close to `(0, 0, 1)`.

## Output

| File | Contents |
| --- | --- |
| `calib/wrist_intrinsics.yaml` | Model, resolution, intrinsics, distortion, focus, errors, `accepted` flag; read by later steps |
| `output/step02/views/view_XX.png` | Each captured image, from 00 |
| `output/step02/views/view_XX_corners.png` | The same image with detected corners |
| `output/step02/raw_sample.png`, `undistorted_sample.png`, `raw_and_undistorted.png` | Last capture, raw and undistorted |

## Pass criteria

- The held-out mean reprojection error (`holdout_mean_error_px`) is at most 0.5 px, so the yaml has `accepted: true` and `calibrate` exits with 0.
- Straight lines look straight in the undistorted image.

If the error is above 0.5 px, the file is still written with `accepted: false` and the command exits with 1. Do not use that file in later steps: add more edge and tilted poses and run `calibrate` again. After passing, keep the resolution, focus, and model unchanged.

## Troubleshooting

- **`No /dev/video* found`, or the camera is listed but cannot be opened**: add your user to the `video` group with `sudo usermod -aG video $USER`, then log out and back in.
- **`Several non-RealSense cameras found`**: pass the wrist camera index with `--device` before the subcommand.
- **`Requested WxH, but the camera delivered ...`**: use a resolution printed by `probe`.
- **`unrecognized arguments: --device`**: move `--device`, `--width`, and `--height` before the subcommand.
- **Fisheye warning about a relaxed condition check**: check the undistorted image; if lines still curve, add more tilted poses and recalibrate.
