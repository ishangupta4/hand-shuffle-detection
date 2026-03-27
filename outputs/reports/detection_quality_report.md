# Keypoint Detection Quality Report

**Videos processed:** 19

## Summary

- Average wrist jitter: 0.0069
- Average flicker frames per video: 104.4

- **Good:** 0 videos
- **Acceptable:** 19 videos
- **Poor:** 0 videos

## Detection Coverage (informational)

Hands leaving the frame is normal during shuffles. These numbers show detection coverage, not quality.

| Video | Frames | Both Hands | Left Only | Right Only | No Hands | Quality |
|-------|--------|-----------|-----------|------------|----------|---------|
| 00001 | 555 | 44.3% | 18.6% | 13.7% | 23.4% | Acceptable |
| 00002 | 325 | 86.5% | 8.9% | 4.0% | 0.6% | Acceptable |
| 00003 | 837 | 56.3% | 21.5% | 14.2% | 8.0% | Acceptable |
| 00004 | 562 | 63.9% | 16.7% | 11.0% | 8.4% | Acceptable |
| 00005 | 595 | 82.2% | 3.5% | 9.6% | 4.7% | Acceptable |
| 00006 | 549 | 73.2% | 3.8% | 9.7% | 13.3% | Acceptable |
| 00007 | 662 | 72.4% | 5.1% | 14.2% | 8.3% | Acceptable |
| 00008 | 537 | 48.0% | 7.6% | 21.4% | 22.9% | Acceptable |
| 00009 | 896 | 51.5% | 15.0% | 18.5% | 15.1% | Acceptable |
| 00010 | 665 | 55.3% | 16.1% | 24.5% | 4.1% | Acceptable |
| 00011 | 669 | 47.8% | 16.9% | 23.5% | 11.8% | Acceptable |
| 00012 | 415 | 42.7% | 11.3% | 26.5% | 19.5% | Acceptable |
| 00013 | 423 | 66.9% | 5.7% | 26.0% | 1.4% | Acceptable |
| 00014 | 376 | 34.3% | 40.4% | 8.5% | 16.8% | Acceptable |
| 00015 | 494 | 40.1% | 26.3% | 9.9% | 23.7% | Acceptable |
| 00016 | 565 | 47.1% | 13.1% | 35.8% | 4.1% | Acceptable |
| 00017 | 729 | 49.5% | 11.9% | 23.7% | 14.8% | Acceptable |
| 00018 | 651 | 59.9% | 8.8% | 19.2% | 12.1% | Acceptable |
| 00019 | 768 | 54.0% | 12.2% | 21.4% | 12.4% | Acceptable |

## Quality Metrics

| Video | Flicker (L) | Flicker (R) | Flicker Total | Wrist Jitter | Quality |
|-------|------------|------------|---------------|-------------|---------|
| 00001 | 73 | 46 | 119 | 0.0061 | Acceptable |
| 00002 | 7 | 15 | 22 | 0.0035 | Acceptable |
| 00003 | 55 | 44 | 99 | 0.0106 | Acceptable |
| 00004 | 51 | 51 | 102 | 0.0105 | Acceptable |
| 00005 | 48 | 9 | 57 | 0.0039 | Acceptable |
| 00006 | 34 | 21 | 55 | 0.0037 | Acceptable |
| 00007 | 26 | 23 | 49 | 0.0071 | Acceptable |
| 00008 | 85 | 23 | 108 | 0.0061 | Acceptable |
| 00009 | 125 | 71 | 196 | 0.0075 | Acceptable |
| 00010 | 75 | 46 | 121 | 0.0061 | Acceptable |
| 00011 | 125 | 94 | 219 | 0.0097 | Acceptable |
| 00012 | 48 | 45 | 93 | 0.0081 | Acceptable |
| 00013 | 57 | 30 | 87 | 0.0068 | Acceptable |
| 00014 | 23 | 9 | 32 | 0.0048 | Acceptable |
| 00015 | 49 | 47 | 96 | 0.0071 | Acceptable |
| 00016 | 71 | 57 | 128 | 0.0102 | Acceptable |
| 00017 | 49 | 40 | 89 | 0.0042 | Acceptable |
| 00018 | 109 | 44 | 153 | 0.0086 | Acceptable |
| 00019 | 115 | 44 | 159 | 0.0073 | Acceptable |

## Terminology

- **Detection coverage:** Percentage of frames where 0, 1, or 2 hands were detected. Hands leaving the frame is expected and does not indicate a problem.
- **Flicker:** Short gaps (1-5 frames) where a hand was detected before and after, but not during. This suggests MediaPipe briefly lost a visible hand — a real quality issue that the cleaning step will interpolate.
- **Wrist jitter:** Median frame-to-frame wrist displacement when the hand IS detected. High jitter means noisy coordinates, often caused by motion blur or low resolution.
- **Quality:** 'Good' = no issues. 'Acceptable' = minor flicker or jitter. 'Poor' = both significant flicker and jitter — consider excluding or reviewing manually.

- 'Left'/'Right' refers to the **subject's** perspective (MediaPipe labels are flipped).
