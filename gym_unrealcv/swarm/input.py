from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GamepadState:
    forward: float = 0.0
    right: float = 0.0
    up: float = 0.0
    yaw: float = 0.0


class GamepadTeleop:
    def __init__(self, deadzone: float = 0.1) -> None:
        self.deadzone = deadzone
        self._pygame: Any | None = None
        self._joystick: Any | None = None

        try:
            import pygame

            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                joystick = pygame.joystick.Joystick(0)
                joystick.init()
                self._pygame = pygame
                self._joystick = joystick
        except Exception:
            self._pygame = None
            self._joystick = None

    @property
    def available(self) -> bool:
        return self._joystick is not None

    def _dz(self, value: float) -> float:
        if abs(value) < self.deadzone:
            return 0.0
        return float(value)

    def poll(self) -> GamepadState:
        if not self.available:
            return GamepadState()

        assert self._pygame is not None
        assert self._joystick is not None

        self._pygame.event.pump()

        left_x = self._dz(self._joystick.get_axis(0))
        left_y = self._dz(self._joystick.get_axis(1))
        right_x = self._dz(self._joystick.get_axis(2))

        up = 0.0
        if self._joystick.get_numaxes() >= 5:
            lt = (self._joystick.get_axis(4) + 1.0) * 0.5
            rt = (self._joystick.get_axis(5) + 1.0) * 0.5 if self._joystick.get_numaxes() > 5 else 0.0
            up = rt - lt

        return GamepadState(
            forward=-left_y,
            right=left_x,
            up=up,
            yaw=right_x,
        )
