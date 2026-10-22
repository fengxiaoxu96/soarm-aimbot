"""Feetech STS3215 position packets.

Goal Position is address 42. Present Position is address 56. Torque Enable is
address 40. Callers decide when torque is switched; reading a position does
not enable it.
"""

from __future__ import annotations

import numpy as np

GOAL_POSITION = 42
PRESENT_POSITION = 56
TORQUE_ENABLE = 40
TICKS_PER_REVOLUTION = 4096
_WRITE = 0x03
_READ = 0x02


def radians_to_tick(angle_rad: float, tick_at_zero: int, sign: int, ticks_per_revolution: int = TICKS_PER_REVOLUTION) -> int:
    if sign not in (-1, 1):
        raise ValueError("Servo direction sign must be 1 or -1")
    tick = int(round(tick_at_zero + sign * float(angle_rad) * ticks_per_revolution / (2.0 * np.pi)))
    if not 0 <= tick <= ticks_per_revolution - 1:
        raise ValueError(f"Joint angle {angle_rad:.4f} rad maps to servo tick {tick}, outside 0 to {ticks_per_revolution - 1}")
    return tick


def tick_to_radians(tick: int, tick_at_zero: int, sign: int, ticks_per_revolution: int = TICKS_PER_REVOLUTION) -> float:
    if sign not in (-1, 1):
        raise ValueError("Servo direction sign must be 1 or -1")
    if not 0 <= int(tick) <= ticks_per_revolution - 1:
        raise ValueError(f"Tick {tick} is outside 0 to {ticks_per_revolution - 1}")
    return sign * (int(tick) - int(tick_at_zero)) * (2.0 * np.pi) / float(ticks_per_revolution)


def sign_from_ticks(tick_at_zero: int, tick_after_positive_move: int) -> int:
    """Positive URDF motion increases ticks when the sign is +1."""

    if int(tick_after_positive_move) == int(tick_at_zero):
        raise ValueError("The tick did not change after moving in the positive direction")
    return 1 if int(tick_after_positive_move) > int(tick_at_zero) else -1


def instruction_packet(servo_id: int, instruction: int, parameters: bytes) -> bytes:
    if not 0 <= servo_id <= 253:
        raise ValueError(f"Invalid servo id {servo_id}")
    body = bytes([servo_id, len(parameters) + 2, instruction]) + bytes(parameters)
    return bytes([0xFF, 0xFF]) + body + bytes([(~sum(body)) & 0xFF])


def goal_position_packet(servo_id: int, tick: int) -> bytes:
    if not 0 <= tick <= TICKS_PER_REVOLUTION - 1:
        raise ValueError(f"Tick {tick} is outside 0 to {TICKS_PER_REVOLUTION - 1}")
    low = tick & 0xFF
    high = (tick >> 8) & 0xFF
    return instruction_packet(servo_id, _WRITE, bytes([GOAL_POSITION, low, high]))


def read_position_packet(servo_id: int) -> bytes:
    return instruction_packet(servo_id, _READ, bytes([PRESENT_POSITION, 2]))


def torque_packet(servo_id: int, enabled: bool) -> bytes:
    return instruction_packet(servo_id, _WRITE, bytes([TORQUE_ENABLE, 1 if enabled else 0]))


def parse_status(buffer: bytes, servo_id: int) -> bytes | None:
    """Return the data bytes of one complete status packet, or None if incomplete.

    A nonzero servo error byte raises. The checksum covers every byte after
    the two header bytes except the checksum itself.
    """

    data = bytes(buffer)
    start = 0
    while start + 6 <= len(data):
        if data[start : start + 2] != b"\xff\xff":
            start += 1
            continue
        if data[start + 2] != servo_id:
            start += 1
            continue
        length = data[start + 3]
        end = start + 4 + length
        if end > len(data):
            return None
        frame = data[start:end]
        if (~sum(frame[2:-1]) & 0xFF) != frame[-1]:
            start += 1
            continue
        error = frame[4]
        if error != 0:
            raise ValueError(f"Servo {servo_id} returned error code {error}")
        return frame[5:-1]
    return None


class Sts3215Bus:
    """One STS/SMS bus. Present position can be read with torque off."""

    def __init__(self, port: str, baud: int = 1000000, timeout_s: float = 0.05) -> None:
        import serial

        self._port = serial.Serial(port, baud, timeout=timeout_s)
        self._timeout_s = float(timeout_s)

    def close(self) -> None:
        self._port.close()

    def read_position(self, servo_id: int) -> int:
        data = self._exchange(read_position_packet(servo_id), servo_id)
        if len(data) != 2:
            raise ValueError(f"Position reply from servo {servo_id} has length {len(data)}, expected 2")
        return data[0] | (data[1] << 8)

    def write_goal(self, servo_id: int, tick: int) -> None:
        self._exchange(goal_position_packet(servo_id, tick), servo_id)

    def set_torque(self, servo_id: int, enabled: bool) -> None:
        self._exchange(torque_packet(servo_id, enabled), servo_id)

    def _exchange(self, packet: bytes, servo_id: int) -> bytes:
        import time

        self._port.reset_input_buffer()
        self._port.write(packet)
        self._port.flush()
        deadline = time.monotonic() + max(self._timeout_s * 4, 0.2)
        buffer = b""
        while time.monotonic() < deadline:
            chunk = self._port.read(32)
            if chunk:
                buffer += chunk
                parsed = parse_status(buffer, servo_id)
                if parsed is not None:
                    return parsed
            elif buffer:
                break
        raise ValueError(f"Servo {servo_id} did not return a complete reply")
