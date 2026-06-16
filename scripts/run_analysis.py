#!/usr/bin/env python
"""
run_analysis.py — CLI entry point for BoulderVision.

Phase 1 supports image analysis (hold detection + color classification).
Video analysis (Phase 3) is wired but marked experimental.

Examples:
    python scripts/run_analysis.py --image data/input/wall.jpg
    python scripts/run_analysis.py --video data/input/climb.mp4 --output data/output/climb_out.mp4
"""

from __future__ import annotations

import argparse
import os
import sys

# Add the project's src/ directory to the import path so we can import the
# library modules regardless of where this script is launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BoulderVision analysis.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a still image to analyze.")
    group.add_argument("--video", help="Path to a video to analyze (Phase 3, experimental).")
    parser.add_argument("--output", help="Output path (video mode).")
    parser.add_argument("--config", help="Optional path to settings.yaml.")
    args = parser.parse_args()

    if args.image:
        import hold_detector
        output = hold_detector.run(args.image, args.config)
        print(f"\nSaved annotated image to: {output}")
    else:
        import video_pipeline
        print("Video mode is experimental (Phase 3 scaffold).")
        timeline = video_pipeline.analyze_video(
            args.video, output_path=args.output, config_path=args.config
        )
        print(f"Processed {len(timeline)} frames.")


if __name__ == "__main__":
    main()
