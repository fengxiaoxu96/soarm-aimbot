# Step 1: Overhead Camera Alignment and Intrinsics

This step aligns the overhead RealSense D455/D455C depth stream to its color stream, saves the color intrinsics to `calib/overhead_intrinsics.yaml`, and checks them by measuring a block of known size. The arm and the wrist camera are not used.

Run all commands from the repository root with the virtual environment activated (see the README).

## What you need

- A RealSense D455 or D455C on a USB 3 port, fixed above the table and looking down. Keep the table about 0.6–1.0 m from the camera; closer than about 0.5 m gives large areas with no depth.
- A matte, opaque, rigid rectangular block (for example a wooden block), at least about 40 mm across the top and 20 mm thick. Measure its top-face width and thickness with calipers.
- A graphical display for `preview`, `calibrate`, and `verify` without ROI flags. `devices`, `self-check`, and `save` work without one.
- Optional: a printed checkerboard on a flat, rigid board, only if the factory intrinsics fail the block check.

Camera options (`--width`, `--height`, `--fps`, `--serial`, `--name-substring`) go **before** the subcommand. The defaults are 848x480 at 30 fps.

## Steps

### 1. Check that the camera is detected

```bash
aimbot-step01 devices
aimbot-step01 self-check
```

`devices` should print a line marked `overhead candidate` whose name contains `455`, with its serial and firmware. `self-check` checks the back-projection math without a camera and should print `Back-projection check passed`.

If several RealSense cameras are connected, pass the serial from `devices`, e.g. `aimbot-step01 --serial <serial> save`. If your overhead camera is not a D455, pass `--name-substring ''`.

### 2. Check the alignment visually

```bash
aimbot-step01 preview
```

The left half is color, the right half is depth aligned to color. The outline of an object with sharp edges should sit in the same place in both halves. Press `s` to save (same as `save` below) or `q` to quit.

### 3. Save the factory intrinsics

```bash
aimbot-step01 save
```

The program discards 30 warm-up frames, takes a temporal median over 15 frames, then a 5x5 spatial median. Do not move the camera or wave a hand in front of it while it runs. It prints the device, resolution, `fx fy cx cy`, distortion, an image-center back-projection, and the valid depth fraction.

### 4. Verify with the block

Lay the block flat near the middle of the image, top face up, with the edge you measured as width running left–right in the image. Leave an empty patch of table next to it. Use your own measurements:

```bash
aimbot-step01 verify --width-mm 40 --height-mm 30
```

- `--width-mm`: top-face edge along the image width.
- `--height-mm`: block thickness, table to top face.
- `--length-mm` (optional): top-face edge along the image height.

The program captures first, then opens two windows on the still image:

1. `step01: top face, then ENTER`: draw a box exactly over the top face. Its left and right sides set the measured width, so they must sit on the physical edges. Do not include any table. Press Enter.
2. `step01: table patch, then ENTER`: draw a box on the table next to the block, not touching it. Press Enter.

Without a display, look at `output/step01/color.png` and pass the boxes as `x,y,w,h` pixels:

```bash
aimbot-step01 verify --width-mm 40 --height-mm 30 \
  --top-roi 360,200,120,80 --table-roi 80,80,60,60
```

### 5. Optional: recalibrate with a checkerboard

Only if the width or thickness check fails and you are sure the boxes were drawn correctly. `--cols`/`--rows` are **inner corners** (a board of 10x7 squares has 9x6 inner corners), and `--square-mm` is the square size measured on the printout (print at 100% scale).

```bash
aimbot-step01 calibrate --cols 9 --rows 6 --square-mm 25
```

Keep the camera fixed and move the board: center, near all four corners and edges, and tilted about 30° in both directions, with the whole pattern in view. Press `SPACE` to capture when the colored corner lines appear, `c` to compute once you have at least 12 views (`--views`), or `q` to abort without changing the file. If the reprojection error is above 0.5 px, add more tilted poses and check `--square-mm`. Then run step 4 again.

Running `save` again later overwrites the checkerboard result with the factory intrinsics.

## Output

| File | Contents |
| --- | --- |
| `calib/overhead_intrinsics.yaml` | Color intrinsics and device info, read by later steps |
| `output/step01/aligned_preview.png` | Color and aligned depth side by side |
| `output/step01/color.png` | Color frame used for the capture |
| `output/step01/depth_m.npy` | Aligned depth, `float32`, meters, 0 = no depth |
| `output/step01/verify.yaml` | Block check results (meters), written by `verify` |

## Pass criteria

- Color and depth edges line up in `aligned_preview.png`.
- `verify` back-projected width is within about 5% of the measured width (`width_error_m` in `verify.yaml`).
- `verify` thickness is within about 10 mm of the measured thickness (`height_error_m` within 0.010 m).

The program prints an extra hint when either limit is exceeded.

## Troubleshooting

- **`No RealSense found`, or the camera is listed but cannot be opened**: use a USB 3 port, then install the udev rules (`sudo apt install librealsense2-udev-rules`) and replug the camera.
- **`Too little valid depth`, or black areas in the depth view**: the table is too close or too far, the lens is blocked, or the surface is shiny, transparent, or black.
- **ROI rejected (too few valid depth points)**: use a larger matte block, or draw a larger box.
- **Thickness is far off**: make sure the table box is on the table next to the block, not on the block or past the table edge.
- **`No graphical display is available`**: run with a display, or pass `--top-roi` and `--table-roi` to `verify`.
