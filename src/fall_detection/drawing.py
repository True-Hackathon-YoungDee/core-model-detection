"""Skeleton overlay drawing, using MediaPipe's own Tasks drawing helpers."""

from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np
from mediapipe.tasks.python import vision

from .fall_fsm import FallState
from .fall_state import FallEvent
from .pose import POSE_CONNECTIONS, PersonPose

logger = logging.getLogger(__name__)

_ALERT_STATES = (FallState.FALL_CONFIRMED, FallState.BED_REST)
_ALERT_COLOR = (0, 0, 255)

_ID_COLORS = (
    (0, 255, 0),
    (255, 128, 0),
    (0, 200, 255),
    (255, 0, 255),
    (0, 255, 255),
    (200, 0, 255),
)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def id_color(person_id: int) -> tuple[int, int, int]:
    return _ID_COLORS[person_id % len(_ID_COLORS)]


def annotate(
    bgr_frame: np.ndarray,
    persons: Sequence[PersonPose],
    hud: str | None = None,
    draw_ids: bool = True,
) -> np.ndarray:
    """Draw skeletons (and optional labels/HUD) onto a copy of ``bgr_frame``."""
    canvas = bgr_frame.copy()
    height, width = canvas.shape[:2]
    style = vision.drawing_styles.get_default_pose_landmarks_style()

    for person in persons:
        vision.drawing_utils.draw_landmarks(
            canvas,
            person.landmarks,
            POSE_CONNECTIONS,
            landmark_drawing_spec=style,
        )
        if not draw_ids:
            continue
        color = id_color(person.person_id)
        x1, y1, x2, y2 = person.bbox_in_pixels(width, height)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
        label = f"#{person.person_id} {person.score:.2f}"
        cv2.putText(canvas, label, (x1, max(y1 - 8, 14)), _FONT, 0.6, color, 2, cv2.LINE_AA)

    if hud:
        cv2.putText(canvas, hud, (10, 24), _FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def annotate_fall_state(
    canvas: np.ndarray,
    events: Sequence[FallEvent],
    *,
    debug: bool = False,
    additional_observation_age_s: float = 0.0,
) -> np.ndarray:
    """Draw a per-person fall-state list and, on any confirmed event, a banner.

    Called after :func:`annotate` on the same canvas it returned -- additive,
    does not touch that function's signature.
    """
    canvas = canvas.copy()
    height, width = canvas.shape[:2]

    y = 50
    for event in events:
        color = _ALERT_COLOR if event.state in _ALERT_STATES else (255, 255, 255)
        label = f"#{event.person_id} {event.state.name}"
        if debug:
            incident = event.incident
            kind = event.alert_kind or (incident.kind if incident is not None else None)
            level = event.evidence_level or (
                incident.evidence_level if incident is not None else None
            )
            stale_age_s = max(
                0.0,
                event.observation_age_s + additional_observation_age_s,
            )
            label += (
                f" {kind.value if kind is not None else '-'}"
                f"/{level.value if level is not None else '-'}"
                f" ev={event.evidence_fraction:.0%}"
                f" {event.evidence_elapsed_s:.1f}/{event.evidence_required_s:.1f}s"
                f" cov={event.coverage_fraction:.0%}"
                f" stale={stale_age_s:.1f}s"
            )
        cv2.putText(canvas, label, (10, y), _FONT, 0.6, color, 2, cv2.LINE_AA)
        y += 24

    confirmed = [event for event in events if event.state in _ALERT_STATES]
    if confirmed:
        banner = "FALL DETECTED" if any(
            event.state == FallState.FALL_CONFIRMED for event in confirmed
        ) else "BED REST"
        cv2.rectangle(canvas, (0, height - 50), (width, height), _ALERT_COLOR, -1)
        text_width = cv2.getTextSize(banner, _FONT, 1.0, 3)[0][0]
        x = max((width - text_width) // 2, 0)
        cv2.putText(canvas, banner, (x, height - 15), _FONT, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

    return canvas
