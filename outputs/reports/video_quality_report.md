# Video Quality Assessment Report

**Total videos:** 19

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total videos | 19 |
| Good | 0 |
| Acceptable | 19 |
| Poor | 0 |
| Unreadable | 0 |
| Average duration | 19.99s |
| Average FPS | 29.7 |

## Per-Video Assessment

| Video | Resolution | FPS | Frames | Duration | Blur (avg) | Brightness (avg) | Rating | Issues |
|-------|-----------|-----|--------|----------|-----------|-----------------|--------|--------|
| 00001.mp4 | 360x640 | 30.0 | 555 | 18.50s | 411.6 | 119.5 | Acceptable | Low resolution (360x640) |
| 00002.mp4 | 360x640 | 30.0 | 325 | 10.83s | 168.0 | 120.1 | Acceptable | Low resolution (360x640) |
| 00003.mp4 | 360x640 | 30.0 | 837 | 27.90s | 327.1 | 134.1 | Acceptable | Low resolution (360x640) |
| 00004.mp4 | 360x640 | 30.0 | 562 | 18.73s | 293.0 | 129.6 | Acceptable | Low resolution (360x640) |
| 00005.mp4 | 360x640 | 30.0 | 595 | 19.83s | 550.2 | 94.8 | Acceptable | Low resolution (360x640) |
| 00006.mp4 | 360x640 | 30.0 | 549 | 18.30s | 315.7 | 154.1 | Acceptable | Low resolution (360x640) |
| 00007.mp4 | 360x640 | 30.0 | 662 | 22.07s | 945.0 | 107.3 | Acceptable | Low resolution (360x640) |
| 00008.mp4 | 360x640 | 30.0 | 537 | 17.90s | 314.8 | 121.6 | Acceptable | Low resolution (360x640) |
| 00009.mp4 | 360x640 | 30.0 | 896 | 29.90s | 542.2 | 111.7 | Acceptable | Low resolution (360x640) |
| 00010.mp4 | 360x640 | 30.0 | 665 | 22.19s | 731.1 | 120.9 | Acceptable | Low resolution (360x640) |
| 00011.mp4 | 360x640 | 30.0 | 669 | 22.32s | 692.8 | 126.7 | Acceptable | Low resolution (360x640) |
| 00012.mp4 | 360x640 | 30.0 | 415 | 13.85s | 786.8 | 90.6 | Acceptable | Low resolution (360x640) |
| 00013.mp4 | 360x640 | 30.0 | 423 | 14.11s | 922.6 | 114.9 | Acceptable | Low resolution (360x640) |
| 00014.mp4 | 360x640 | 30.0 | 376 | 12.55s | 780.0 | 103.7 | Acceptable | Low resolution (360x640) |
| 00015.mp4 | 360x640 | 30.0 | 494 | 16.48s | 427.7 | 122.8 | Acceptable | Low resolution (360x640) |
| 00016.mp4 | 360x640 | 25.0 | 565 | 22.60s | 797.6 | 126.4 | Acceptable | Low resolution (360x640) |
| 00017.mp4 | 360x640 | 30.0 | 729 | 24.32s | 1205.5 | 113.8 | Acceptable | Low resolution (360x640) |
| 00018.mp4 | 360x640 | 30.0 | 651 | 21.72s | 757.6 | 114.2 | Acceptable | Low resolution (360x640) |
| 00019.mp4 | 360x640 | 30.0 | 768 | 25.63s | 403.1 | 110.4 | Acceptable | Low resolution (360x640) |

## Recommendations

All videos meet minimum quality standards. No exclusions recommended.

## Notes

- Blur is measured via Laplacian variance (higher = sharper). Threshold: <50 is poor, >100 is good.
- Brightness is mean pixel intensity (0-255). Range 40-220 is acceptable.
- Sample frames are saved in `outputs/sample_frames/` for manual inspection.
- Hand visibility and occlusion should be verified manually using the sample frames.
