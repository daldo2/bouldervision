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
import numpy as np

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


@dataclass
class Move:
    """One climbing move: a limb arriving on a (new) hold."""
    order: int          # 1-based move number in the climb
    frame_index: int
    time_s: float
    limb: str           # left_hand / right_hand / left_foot / right_foot
    hold: int           # hold index the limb moved ONTO
    start: bool = False # part of the initial established position


def extract_moves(timeline: List["FrameAnalysis"], fps: float = 30.0,
                  min_hold_frames: int = 5) -> List["Move"]:
    """Turn a per-frame contact timeline into an ordered move sequence ("beta").

    A move is recorded when a limb settles on a hold *different* from the one it
    last held — i.e. it grabbed something new. A limb releasing (going to None)
    does NOT reset its memory, so letting go and re-gripping the SAME hold is not
    counted as a move. To avoid logging a hold the climber only brushed, a new
    hold must be held for at least `min_hold_frames` frames to count.

    The first time each limb settles is flagged `start` (the starting position).
    Returns moves ordered by time.
    """
    # How long each (limb, hold) contact run lasts, so we can ignore brief ones.
    moves: List[Move] = []
    last_hold = {limb: None for limb in ("left_hand", "right_hand", "left_foot", "right_foot")}
    started = set()
    n = len(timeline)

    def run_length(t: int, limb: str, hold: int) -> int:
        length = 0
        while t + length < n and timeline[t + length].contacts.get(limb) == hold:
            length += 1
        return length

    order = 0
    for t, frame in enumerate(timeline):
        for limb in last_hold:
            hold = frame.contacts.get(limb)
            if hold is None or hold == last_hold[limb]:
                continue
            if run_length(t, limb, hold) < min_hold_frames:
                continue  # too brief — a brush, not a move
            order += 1
            is_start = limb not in started
            started.add(limb)
            moves.append(Move(order=order, frame_index=frame.frame_index,
                              time_s=frame.frame_index / fps, limb=limb,
                              hold=hold, start=is_start))
            last_hold[limb] = hold
    return moves


_LIMB_SHORT = {"left_hand": "LH", "right_hand": "RH", "left_foot": "LF", "right_foot": "RF"}


def print_moves(moves: List["Move"]) -> None:
    """Print the move sequence (beta) in a readable form."""
    print("-" * 40)
    if not moves:
        print("No moves detected.")
        return
    starts = [m for m in moves if m.start]
    seq = [m for m in moves if not m.start]
    print("Move sequence (beta):")
    if starts:
        pos = "  ".join(f"{_LIMB_SHORT[m.limb]}->#{m.hold}" for m in starts)
        print(f"  start ({starts[-1].time_s:4.1f}s):  {pos}")
    for i, m in enumerate(seq, start=1):
        print(f"  {i:2d}. {m.time_s:5.1f}s  {_LIMB_SHORT[m.limb]} -> #{m.hold}")
    print(f"  ({len(seq)} moves after establishing)")


def analyze_video(
    video_path: str,
    output_path: Optional[str] = None,
    config_path: Optional[str] = None,
    detect_every: int = 0,
    detect_frame: int = 0,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    stabilize: bool = False,
    route_color: Optional[str] = None,
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

    # Load both models once, lazily (torch is heavy and only needed here).
    import hold_detector
    import pose_estimator

    detector = hold_detector.load_detector(config["models"]["hold_detector"])
    pose_model = pose_estimator.load_pose_model(config["models"]["pose_estimator"])
    det = config["detection"]
    pose_cfg = config["pose"]
    reach = pose_cfg.get("reach_frac", 0.33)
    max_speed = pose_cfg.get("contact_max_speed_px", 12)

    def detect_at(frame) -> List[tuple]:
        boxes = hold_detector.detect_holds(
            detector, frame, det["confidence"], det["iou"], det["max_detections"]
        )
        return [(x1, y1, x2, y2) for (x1, y1, x2, y2, _conf) in boxes]

    # Detect the static holds once on a chosen (ideally empty-wall) frame, so the
    # hold indices are stable for the whole clip.
    hold_boxes: List[tuple] = []
    ref_boxes: List[tuple] = []
    stabilizer = None
    if detect_every == 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, detect_frame)
        ok, dframe = cap.read()
        if ok:
            hold_boxes = detect_at(dframe)
            if route_color:
                hold_boxes = _filter_by_color(dframe, hold_boxes, route_color, config)
            ref_boxes = list(hold_boxes)
            if stabilize:
                import stabilize as stab  # lazy: only needed for handheld clips
                stabilizer = stab.CameraStabilizer(dframe)

    # PASS 1 — collect per-frame holds + pose (no contacts yet; they need the
    # whole sequence for temporal smoothing).
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    processed = 0
    per_frame: List[dict] = []  # {index, boxes, pose|None}
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if detect_every > 0 and frame_index % detect_every == 0:
                hold_boxes = detect_at(frame)
            elif stabilizer is not None:
                hold_boxes = stabilizer.warp_boxes(ref_boxes, frame)
            poses = pose_estimator.estimate_pose(pose_model, frame, pose_cfg["confidence"])
            per_frame.append({
                "index": frame_index,
                "boxes": list(hold_boxes),
                "pose": poses[0] if poses else None,
            })
            frame_index += 1
            processed += 1
            if max_frames is not None and processed >= max_frames:
                break
    finally:
        cap.release()

    # Temporal keypoint smoothing (steadies jitter, e.g. during a wide split).
    window = pose_cfg.get("smooth_window", 1)
    if window and window > 1 and per_frame:
        blank = np.zeros((17, 3), dtype=float)
        stacked = [pose_estimator.FramePose(m["pose"].keypoints if m["pose"] is not None else blank)
                   for m in per_frame]
        smoothed = pose_estimator.smooth_keypoint_sequence(stacked, window)
        for m, sp in zip(per_frame, smoothed):
            if m["pose"] is not None:
                m["pose"] = sp  # keep gaps (no pose) as gaps

    # Extremity (hand/foot) points per frame.
    points_per_frame: List[Optional[dict]] = [
        pose_estimator.limb_points(m["pose"], reach) if m["pose"] is not None else None
        for m in per_frame
    ]
    boxes_per_frame = [m["boxes"] for m in per_frame]

    # Sticky, stateful contact resolution (engage when settled, stay while near).
    raw_contacts = pose_estimator.resolve_contact_sequence(
        points_per_frame, boxes_per_frame,
        pose_cfg["touch_distance_px"], pose_cfg.get("release_distance_px", 70),
        max_speed, pose_cfg["confidence"],
    )

    # Light hysteresis cleanup: bridge any residual dropouts, drop 1-frame flicker.
    gap = pose_cfg.get("contact_gap_frames", 8)
    min_run = pose_cfg.get("contact_min_run", 2)
    contacts_seq = [dict(c) for c in raw_contacts]
    for limb in pose_estimator.CONTACT_LIMBS:
        cleaned = pose_estimator.smooth_contact_sequence(
            [c[limb] for c in raw_contacts], gap, min_run)
        for t, v in enumerate(cleaned):
            contacts_seq[t][limb] = v

    timeline = [FrameAnalysis(frame_index=m["index"], contacts=contacts_seq[t])
                for t, m in enumerate(per_frame)]

    # PASS 2 — draw the annotated video using the smoothed contacts.
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        try:
            for t, m in enumerate(per_frame):
                ok, frame = cap.read()
                if not ok:
                    break
                poses = [m["pose"]] if m["pose"] is not None else []
                _draw_frame(frame, m["boxes"], poses, contacts_seq[t], points_per_frame[t])
                writer.write(frame)
        finally:
            cap.release()
            writer.release()

    return timeline


def print_summary(timeline: List["FrameAnalysis"], primary_min_frames: int = 30) -> None:
    """Print a human-readable contact summary for a finished timeline.

    `primary_min_frames` is the dwell threshold for a hold to count as a real
    "route hold" (vs. a fleeting near-pass) — set it from fps (e.g. 1 second).
    """
    primary = holds_used(timeline, min_frames=primary_min_frames)
    per_limb = summarize_contacts(timeline)
    print("-" * 40)
    print(f"Analyzed {len(timeline)} frames")
    contact_frames = sum(1 for f in timeline if any(v is not None for v in f.contacts.values()))
    print(f"Frames with >=1 limb gripping: {contact_frames}")
    print(f"Route holds (held >= {primary_min_frames} frames): {len(primary)}  {sorted(primary)}")
    for limb in ("left_hand", "right_hand", "left_foot", "right_foot"):
        # Only show this limb's holds that clear the dwell threshold.
        top = [(i, n) for i, n in per_limb.get(limb, Counter()).most_common() if n >= primary_min_frames]
        pretty = ", ".join(f"hold#{i}:{n}f" for i, n in top[:5]) or "—"
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
    parser.add_argument("--stabilize", action="store_true",
                        help="Handheld camera: warp the static holds to follow the camera each frame.")
    parser.add_argument("--route-color", default=None,
                        help="Analyze only one route: keep holds of this color (red/blue/green/...).")
    args = parser.parse_args()

    timeline = analyze_video(
        args.video, output_path=args.out, config_path=args.config,
        detect_every=args.detect_every, detect_frame=args.detect_frame,
        start_frame=args.start_frame, max_frames=args.max_frames, stabilize=args.stabilize,
        route_color=args.route_color,
    )
    # Translate the "real route hold" dwell threshold from seconds to frames.
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    cfg = utils.load_config(args.config) if args.config else utils.load_config()
    min_frames = max(1, round(cfg["pose"].get("primary_min_seconds", 1.0) * fps))
    print_summary(timeline, primary_min_frames=min_frames)
    print_moves(extract_moves(timeline, fps, min_hold_frames=max(3, round(0.2 * fps))))
    if args.out:
        print(f"Annotated video: {args.out}")


def _filter_by_color(frame, boxes, color, config, include_volumes=True):
    """Keep one route's holds: those matching `color`, PLUS shared volumes.

    A route's hand-holds are one color, but climbers still stand on neutral
    volumes — so by default we also keep big boxes (volumes, by area) regardless
    of colour, so footholds on volumes still register. Uses the Phase-2 colour
    machinery; prints which colours are present so the user can pick a valid one.
    """
    import route_extractor as rex
    refs = utils.reference_labs(config["draw_colors"])
    chroma_min = config.get("color_naming", {}).get("chroma_min", 12)
    vol_frac = config.get("filter", {}).get("volume_area_frac", 0.04)
    img_area = float(frame.shape[0] * frame.shape[1])

    holds = rex.build_holds(frame, [(*b, 1.0) for b in boxes])
    present, kept, vols = Counter(), [], 0
    for h in holds:
        name = utils.nearest_color_name(h.lab, refs, chroma_min)
        present[name] += 1
        x1, y1, x2, y2 = h.box
        is_volume = include_volumes and ((x2 - x1) * (y2 - y1) / img_area) >= vol_frac
        if name == color:
            kept.append(h.box)
        elif is_volume:
            kept.append(h.box)
            vols += 1
    print(f"     route filter: colors on wall = {dict(present.most_common())}")
    print(f"     route filter: keeping {len(kept) - vols} '{color}' holds + {vols} volumes "
          f"(of {len(boxes)} detected)")
    return kept


def _draw_frame(frame, hold_boxes, poses, contacts, points=None) -> None:
    """Overlay holds (touched ones highlighted), keypoints, and contacts in place."""
    touched = {idx for idx in contacts.values() if idx is not None}
    for i, (x1, y1, x2, y2) in enumerate(hold_boxes):
        color = (0, 255, 0) if i in touched else (160, 160, 160)
        thickness = 2 if i in touched else 1   # thin borders; big boxes looked heavy at 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    for pose in poses:
        for x, y, conf in pose.keypoints:
            if conf > 0:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 200, 255), -1)

    # The estimated hand/foot contact points: green when gripping, red otherwise.
    if points:
        for limb, (x, y, c) in points.items():
            if c <= 0:
                continue
            gripping = contacts.get(limb) is not None
            cv2.circle(frame, (int(x), int(y)), 7, (0, 255, 0) if gripping else (0, 0, 255), 2)


if __name__ == "__main__":
    main()
