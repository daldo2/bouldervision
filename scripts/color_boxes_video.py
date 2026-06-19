"""Render a video where each detected hold gets a box OUTLINE in its detected
color + the color name. No route grouping, no pose — just per-hold color, so you
can eyeball how color naming behaves across a moving shot of the wall.

    python scripts/color_boxes_video.py data/input/new_videos/<vid>.mp4 \
        --out data/output/color_boxes.mp4 --start 0 --duration 60 --height 1280 --fps 12
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import detection_filter as dfilt  # noqa: E402
from src import hold_detector as hd  # noqa: E402
from src import route_extractor as rex  # noqa: E402
from src import utils  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="data/output/color_boxes.mp4")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--fps", type=float, default=12.0)
    args = ap.parse_args()

    cfg = utils.load_config()
    det = cfg["detection"]
    model = hd.load_detector(cfg["models"]["hold_detector"])
    refs = utils.reference_labs(cfg["draw_colors"], cfg.get("color_naming", {}).get("hue_anchors"))
    cmin = cfg.get("color_naming", {}).get("chroma_min", 10)
    draw_colors = cfg["draw_colors"]

    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / args.fps)))
    start_f = int(args.start * src_fps)
    end_f = int((args.start + args.duration) * src_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    writer = None
    fi, written = start_f, 0
    while fi < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        if (fi - start_f) % step == 0:
            h, w = frame.shape[:2]
            scale = args.height / h
            frame = cv2.resize(frame, (int(w * scale), args.height))
            H, W = frame.shape[:2]

            # quiet detection (no per-frame prints)
            res = model(frame, conf=det["confidence"], iou=det["iou"],
                        max_det=det["max_detections"], verbose=False)
            raw = []
            for b in res[0].boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                raw.append((int(x1), int(y1), int(x2), int(y2),
                            float(b.conf[0]), int(b.cls[0])))
            dets = dfilt.classify_detections(raw, frame, cfg["filter"])
            holds = rex.build_holds(frame, dfilt.holds_only(dets), use_mask=False)

            for hd_ in holds:
                name = utils.nearest_color_name(hd_.lab, refs, cmin)
                col = utils.draw_color_for(name, draw_colors)
                x1, y1, x2, y2 = hd_.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3, cv2.LINE_AA)
                cv2.putText(frame, name, (x1, max(14, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, name, (x1, max(14, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if writer is None:
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (W, H))
            writer.write(frame)
            written += 1
            if written % 25 == 0:
                print(f"  {written} frames written (src frame {fi})", flush=True)
        fi += 1

    cap.release()
    if writer:
        writer.release()
    print(f"done: {written} frames -> {args.out}")


if __name__ == "__main__":
    main()
