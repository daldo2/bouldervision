"""BoulderVision — bouldering video analysis.

Package modules:
    hold_detector   — detect holds and classify them by color (Phase 1)
    route_extractor — group holds into routes by color + position (Phase 2)
    pose_estimator  — track the climber's skeleton across frames (Phase 3)
    video_pipeline  — orchestrate full-video analysis (Phase 3+)
    utils           — shared config / color / drawing / I/O helpers
"""

__version__ = "0.1.0"
