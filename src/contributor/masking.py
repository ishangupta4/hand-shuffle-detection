"""Hand bounding box computation and pixel masking for privacy-preserving video."""

import numpy as np


class Masker:
    def __init__(self, padding_px):
        self.padding_px = padding_px

    def apply(self, frame, keypoints, hand_mask):
        """Zero out all pixels outside detected hand bounding boxes.

        Args:
            frame: (H, W, 3) RGB uint8
            keypoints: (2, 21, 3) normalized [0, 1] landmarks; index 0=left, 1=right
            hand_mask: (2,) bool — True if hand was detected

        Returns:
            (H, W, 3) masked frame; all-black if no hands detected
        """
        H, W = frame.shape[:2]
        output = np.zeros_like(frame)

        for h in range(2):
            if not hand_mask[h]:
                continue
            lms = keypoints[h]  # (21, 3)
            xs = lms[:, 0] * W
            ys = lms[:, 1] * H
            x1 = max(0, int(xs.min()) - self.padding_px)
            y1 = max(0, int(ys.min()) - self.padding_px)
            x2 = min(W, int(xs.max()) + self.padding_px)
            y2 = min(H, int(ys.max()) + self.padding_px)
            output[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

        return output
