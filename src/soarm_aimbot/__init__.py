"""soarm_aimbot: calibrate an SO-101 arm with an overhead and a wrist camera, then aim at table objects.

Layers, and the only direction they may import:

- ``soarm_aimbot.geometry`` has no hardware and no robot code.
- ``soarm_aimbot.cameras`` adapts a specific camera and may use geometry.
- ``soarm_aimbot.robots`` adapts the position-servo bus and does not import steps.
- ``soarm_aimbot.steps`` holds one runnable command each. A step imports the
  cameras, robots, and geometry it needs, and does not import other steps.
"""

__version__ = "0.1.0"
