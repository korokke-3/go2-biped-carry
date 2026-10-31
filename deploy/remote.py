"""Unitree wireless remote in LowState.wireless_remote (40 bytes, unitree_sdk2 xRockerBtnDataStruct):
2 header bytes, 16 key bits (R1 L1 start select R2 L2 F1 F2 A B X Y up right down left), then float32 lx, rx, ry, L2, ly.
encode() writes the same layout, so the DDS simulator (deploy/sim_dds.py) presses buttons the way the remote does.
"""
from __future__ import annotations

import struct

BUTTONS = ("R1", "L1", "start", "select", "R2", "L2", "F1", "F2", "A", "B", "X", "Y", "up", "right", "down", "left")


def decode(raw) -> tuple[frozenset, dict]:
    buf = bytes(bytearray(raw))
    keys = struct.unpack_from("<H", buf, 2)[0]
    lx, rx, ry, _l2, ly = struct.unpack_from("<5f", buf, 4)
    return frozenset(b for i, b in enumerate(BUTTONS) if keys >> i & 1), {"lx": lx, "ly": ly, "rx": rx, "ry": ry}


def encode(buttons=(), lx=0.0, ly=0.0, rx=0.0, ry=0.0) -> list:
    keys = 0
    for b in buttons:
        keys |= 1 << BUTTONS.index(b)
    buf = bytearray(40)
    struct.pack_into("<H", buf, 2, keys)
    struct.pack_into("<5f", buf, 4, lx, rx, ry, 0.0, ly)
    return list(buf)


class Edges:
    """Buttons on the press, not while held."""

    def __init__(self):
        self.prev = frozenset()

    def update(self, buttons: frozenset) -> frozenset:
        new = buttons - self.prev
        self.prev = buttons
        return new


def sticks_to_command(sticks: dict, max_fwd: float, max_back: float, max_vy: float, max_wz: float, dz: float = 0.05):
    """(vx, vy, wz), body frame (x forward, y left, wz > 0 turns left): left stick up/down = vx, left/right = vy,
    right stick left/right = wz. Sticks within dz of the centre read 0."""
    def f(v):
        return 0.0 if abs(v) < dz else max(-1.0, min(1.0, v))
    ly, lx, rx = f(sticks["ly"]), f(sticks["lx"]), f(sticks["rx"])
    return (ly * (max_fwd if ly > 0 else max_back), -lx * max_vy, -rx * max_wz)
