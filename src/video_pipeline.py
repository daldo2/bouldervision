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
from collections import Counter, defaultdict
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


def summarize_contacts(timeline: List["FrameAnalysis"]) -> Dict[str, Counter]:
    """Aggregate a per-frame timeline into per-limb hold usage.

    Pure post-processing (no model needed): for each contact limb, counts how
    many frames it spent on each hold index. This is the raw material for "which
    holds did the climber use, and for how long" and feeds Phase 4 force work.

    Returns {limb: Counter({hold_index: frame_count, ...})}. Frames where a limb
    touched nothing are not counted.
    """
    per_limb: Dict[str, Counter] = defaultdict(Counter)
    for frame in timeline:
        for limb, hold_index in frame.contacts.items():
            if hold_index is not None:
                per_limb[limb][hold_index] += 1
    return dict(per_limb)


def holds_used(timeline: List["FrameAnalysis"], min_frames: int = 1) -> set:
    """Set of hold indices touched by any limb for at least `min_frames` frames.

    Filtering by `min_frames` drops fleeting brush-pasts so we keep only holds
    the climber actually used.
    """
    counts: Counter = Counter()
    for limb_counter in summarize_contacts(timeline).values():
        counts.update(limb_counter)
    return {idx for idx, n in counts.items() if n >= min_frames}


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

    # Load both models once, lazily (torch is heavy and only needed here).
    import hold_detector
    import pose_estimator

    detector = hold_detector.load_detector(config["models"]["hold_detector"])
    pose_model = pose_estimator.load_pose_model(config["models"]["pose_estimator"])
    det = config["detection"]
    pose_cfg = config["pose"]

    hold_boxes: List[tuple] = []  # static holds, refreshed every `detect_every` frames

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # end of video

            # Holds don't move, so only re-detect periodically (cheap on long clips).
            if frame_index % detect_every == 0:
                boxes = hold_detector.detect_holds(
                    detector, frame, det["confidence"], det["iou"], det["max_detections"]
                )
                hold_boxes = [(x1, y1, x2, y2) for (x1, y1, x2, y2, _conf) in boxes]

            # Estimate the climber's pose and work out limb→hold contacts.
            poses = pose_estimator.estimate_pose(pose_model, frame, pose_cfg["confidence"])
            contacts = {limb: None for limb in pose_estimator.CONTACT_LIMBS}
            if poses:
                contacts = pose_estimator.touched_holds(
                    poses[0],  # assume the most prominent person is the climber
                    hold_boxes,
                    pose_cfg["touch_distance_px"],
                    pose_cfg["confidence"],
                )

            if writer is not None:
                _draw_frame(frame, hold_boxes, poses, contacts)
                writer.write(frame)

            timeline.append(FrameAnalysis(frame_index=frame_index, contacts=contacts))
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return timeline


def _draw_frame(frame, hold_boxes, poses, contacts) -> None:
    """Overlay holds (touched ones highlighted), keypoints, and contacts in place."""
    touched = {idx for idx in contacts.values() if idx is not None}
    for i, (x1, y1, x2, y2) in enumerate(hold_boxes):
        color = (0, 255, 0) if i in touched else (160, 160, 160)
        thickness = 3 if i in touched else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    for pose in poses:
        for x, y, conf in pose.keypoints:
            if conf > 0:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 200, 255), -1)
