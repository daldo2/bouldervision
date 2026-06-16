"""
video_pipeline.py — Phase 3+ orchestration (scaffold).

Ties the pieces together for a full video:

    open video ──▶ for each frame:
        detect holds (once, or every N frames since holds don't move)
        estimate climber pose
        compute which holds each limb is touching
      ──▶ write an annotated frame to the output video
      ──▶ accumulate a per-frame timeline for later analysis (Phase 4 force est.)

This is a scaffold. The frame loop is sketched; the per-frame analysis calls
into hold_detector / pose_estimator and is left as Phase 3 TODOs.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils  # noqa: E402


@dataclass
class FrameAnalysis:
    """Everything we learned about a single video frame."""
    frame_index: int
    # limb -> hold index being touched (or None). Populated in Phase 3.
    contacts: Dict[str, Optional[int]] = field(default_factory=dict)


def analyze_video(
    video_path: str,
    output_path: Optional[str] = None,
    config_path: Optional[str] = None,
    detect_every: int = 30,
) -> List[FrameAnalysis]:
    """Run the full analysis pipeline over a video file.

    Args:
        video_path: input video.
        output_path: where to write the annotated video (optional).
        config_path: optional settings.yaml override.
        detect_every: re-run hold detection every N frames. Holds are static,
            so we don't need to detect them on every single frame — this saves
            a lot of compute on long videos.

    Returns:
        A per-frame timeline (list of FrameAnalysis).
    """
    config = utils.load_config(config_path) if config_path else utils.load_config()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    # Read basic video properties so we can write a matching output video.
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    timeline: List[FrameAnalysis] = []
    frame_index = 0

    # TODO(phase-3): load the hold detector and pose model once, before the loop:
    #   detector = hold_detector.load_detector(config["models"]["hold_detector"])
    #   pose_model = pose_estimator.load_pose_model(config["models"]["pose_estimator"])

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # end of video

            # TODO(phase-3): on every `detect_every`th frame, re-detect holds.
            # TODO(phase-3): estimate pose on this frame.
            # TODO(phase-3): compute limb→hold contacts via
            #                pose_estimator.touched_holds(...).
            # TODO(phase-3): draw skeleton + highlight touched holds on `frame`.

            timeline.append(FrameAnalysis(frame_index=frame_index))

            if writer is not None:
                writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return timeline
