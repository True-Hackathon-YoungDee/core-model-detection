"""How a frame becomes poses.

Kept in its own module so the CLI can build its parser without importing
mediapipe -- logging has to be configured before that happens.
"""

from __future__ import annotations

from enum import Enum


class Strategy(str, Enum):
    """Which detection path to run.

    ``NATIVE`` hands the whole frame to one landmarker. Cheap, and correct for a
    single subject -- but BlazePose proposes one dominant region per frame, so
    with several people in view it reliably reports only one of them.

    ``CASCADE`` runs a person detector first and gives each body its own crop,
    which costs roughly twice as much and finds everybody.

    ``AUTO`` picks NATIVE at ``num_poses == 1`` and CASCADE above it.
    """

    AUTO = "auto"
    NATIVE = "native"
    CASCADE = "cascade"

    def __str__(self) -> str:
        return self.value
