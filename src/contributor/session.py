"""ContributorSession dataclass — holds all per-session state."""

from dataclasses import dataclass, field


@dataclass
class ContributorSession:
    session_id: str
    video_id: str
    consent_id: str = ""
    collection_mode: str = "both"
    frames: list = field(default_factory=list)
    keypoints: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    frame_count: int = 0
    recording: bool = True
    finalized: bool = False
