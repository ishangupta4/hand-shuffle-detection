"""Frame buffering and masked video assembly."""

import cv2
import numpy as np


# Codecs tried in order for browser-compatible video output.
# avc1 = H.264 (best for browsers, uses AVFoundation on macOS).
# mp4v = MPEG-4 Part 2 (fallback — works but some browsers reject it).
_PREVIEW_CODECS = ['avc1', 'H264', 'mp4v']


class Recorder:
    def __init__(self, cfg):
        self.cfg = cfg

    def add_frame(self, session, frame_rgb, keypoints, hand_mask):
        session.keypoints.append(keypoints.copy())
        session.masks.append(hand_mask.copy())
        if self.cfg.collection_mode in ("video_only", "both") and frame_rgb is not None:
            session.frames.append(frame_rgb.copy())
        session.frame_count += 1

    def build_npy_arrays(self, session):
        if not session.keypoints:
            return (
                np.zeros((0, 2, 21, 3), dtype=np.float32),
                np.zeros((0, 2), dtype=bool),
            )
        kp = np.stack(session.keypoints, axis=0).astype(np.float32)  # (T, 2, 21, 3)
        mk = np.stack(session.masks, axis=0)                          # (T, 2)
        return kp, mk

    def build_masked_video(self, session, output_path, masker):
        if not session.frames:
            return None

        w, h = self.cfg.recording.resolution
        actual_fps = session.frame_count / max(self.cfg.recording.duration_seconds, 1)
        actual_fps = max(1.0, actual_fps)

        writer = self._open_writer(output_path, actual_fps, w, h)
        if writer is None:
            print(f"  VideoWriter: all codecs failed for {output_path}")
            return None

        for i, frame_rgb in enumerate(session.frames):
            kp = session.keypoints[i] if i < len(session.keypoints) else np.zeros((2, 21, 3), dtype=np.float32)
            mk = session.masks[i] if i < len(session.masks) else np.zeros(2, dtype=bool)
            masked = masker.apply(frame_rgb, kp, mk)
            resized = cv2.resize(masked, (w, h))
            bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            writer.write(bgr)

        writer.release()
        return output_path

    def _open_writer(self, output_path, fps, w, h):
        """Try codecs in order; return the first VideoWriter that opens."""
        for codec in _PREVIEW_CODECS:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
                if writer.isOpened():
                    print(f"  VideoWriter: using codec '{codec}'")
                    return writer
                writer.release()
            except Exception as e:
                print(f"  VideoWriter: codec '{codec}' failed: {e}")
        return None
