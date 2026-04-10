"""Orchestrates recorder → masking → storage → labels for a contributor session."""

import datetime
import threading
from pathlib import Path

from .masking import Masker
from .recorder import Recorder
from .storage import make_storage


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.storage = make_storage(cfg)
        self._recorder = Recorder(cfg)

    def add_frame(self, session, frame_rgb, keypoints, hand_mask):
        self._recorder.add_frame(session, frame_rgb, keypoints, hand_mask)

    def save_session(self, session, start_hand, end_hand):
        kp_array, mask_array = self._recorder.build_npy_arrays(session)

        meta = {
            "session_id": session.session_id,
            "consent_id": session.consent_id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "frame_count": session.frame_count,
            "collection_mode": session.collection_mode,
        }

        mode = self.cfg.collection_mode

        if mode in ("npy_only", "both"):
            self.storage.save_keypoints(session.video_id, kp_array, mask_array, meta)

        if mode in ("video_only", "both"):
            preview = Path(session.preview_path) if session.preview_path else None
            if preview and preview.exists() and preview.stat().st_size > 0:
                # Reuse the pre-built preview — just move it, no rebuild needed.
                self.storage.save_video(session.video_id, preview)
            elif session.frames:
                # Fallback: build from frames if preview was never created.
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                masker = Masker(self.cfg.recording.hand_mask_padding_px)
                self._recorder.build_masked_video(session, tmp_path, masker)
                self.storage.save_video(session.video_id, tmp_path)
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

        self.storage.save_labels(
            session.video_id,
            start_hand,
            end_hand,
            session.session_id,
            datetime.datetime.utcnow().isoformat() + "Z",
        )

    def save_session_async(self, session, start_hand, end_hand):
        t = threading.Thread(
            target=self.save_session,
            args=(session, start_hand, end_hand),
            daemon=True,
        )
        t.start()
        return t
