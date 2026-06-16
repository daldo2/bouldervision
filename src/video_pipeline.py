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
    detect_every: int = 0,
    detect_frame: int = 0,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
) -> List[FrameAnalysis]:
    """Run the full analysis pipeline over a video file.

    Args:
        video_path: input video.
        output_path: where to write the annotated video (optional).
        config_path: optional settings.yaml override.
        detect_every: re-run hold detection every N frames. 0 (default) means
            detect ONCE (at `detect_frame`) and reuse — holds are static and the
            box *indices* must stay stable for the contact summary to mean
            anything. Use a positive value only with hold tracking (TODO) or a
            moving camera.
        detect_frame: which frame to detect holds on (use an empty-wall frame to
            avoid the climber occluding holds).
        start_frame: first frame to analyze (skip a long empty intro).
        max_frames: stop after this many analyzed frames (None = whole clip).

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

    # Load both models once, lazily (torch is heavy and only needed here).
    import hold_detector
    import pose_estimator

    detector = hold_detector.load_detector(config["models"]["hold_detector"])
    pose_model = pose_estimator.load_pose_model(config["models"]["pose_estimator"])
    det = config["detection"]
    pose_cfg = config["pose"]

    def detect_at(frame) -> List[tuple]:
        boxes = hold_detector.detect_holds(
            detector, frame, det["confidence"], det["iou"], det["max_detections"]
        )
        return [(x1, y1, x2, y2) for (x1, y1, x2, y2, _conf) in boxes]

    # Detect the static holds once on a chosen (ideally empty-wall) frame, so the
    # hold indices are stable for the whole clip.
    hold_boxes: List[tuple] = []
    if detect_every == 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, detect_frame)
        ok, dframe = cap.read()
        if ok:
            hold_boxes = detect_at(dframe)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    processed = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # end of video

            # Moving-camera mode: periodically re-detect (indices not stable).
            if detect_every > 0 and frame_index % detect_every == 0:
                hold_boxes = detect_at(frame)

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
            processed += 1
            if max_frames is not None and processed >= max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return timeline


def print_summary(timeline: List["FrameAnalysis"], min_frames: int = 3) -> None:
    """Print a human-readable contact summary for a finished timeline."""
    used = holds_used(timeline, min_frames=min_frames)
    per_limb = summarize_contacts(timeline)
    print("-" * 40)
    print(f"Analyzed {len(timeline)} frames")
    contact_frames = sum(1 for f in timeline if any(v is not None for v in f.contacts.values()))
    print(f"Frames with >=1 limb on a hold: {contact_frames}")
    print(f"Distinct holds used (>= {min_frames} frames): {len(used)}  {sorted(used)}")
    for limb in ("left_hand", "right_hand", "left_foot", "right_foot"):
        top = per_limb.get(limb, Counter()).most_common(4)
        pretty = ", ".join(f"hold#{i}:{n}f" for i, n in top) or "—"
        print(f"  {limb:11} {pretty}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Analyze a climbing video: pose + limb→hold contacts.")
    parser.add_argument("video", help="Path to the input video.")
    parser.add_argument("--out", default=None, help="Write an annotated video here.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--detect-frame", type=int, default=0,
                        help="Frame to detect the (static) holds on — pick an empty-wall frame.")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame to analyze.")
    parser.add_argument("--max-frames", type=int, default=None, help="Analyze at most N frames.")
    parser.add_argument("--detect-every", type=int, default=0,
                        help="Moving camera: re-detect holds every N frames (default 0 = once).")
    args = parser.parse_args()

    timeline = analyze_video(
        args.video, output_path=args.out, config_path=args.config,
        detect_every=args.detect_every, detect_frame=args.detect_frame,
        start_frame=args.start_frame, max_frames=args.max_frames,
    )
    print_summary(timeline)
    if args.out:
        print(f"Annotated video: {args.out}")


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


if __name__ == "__main__":
    main()
