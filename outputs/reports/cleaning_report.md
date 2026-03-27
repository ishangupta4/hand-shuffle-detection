# Keypoint Cleaning Report

**Videos cleaned:** 19

## Cleaning Philosophy

Only two types of issues are corrected:

1. **Flicker gaps** -- short dropouts (1-5 frames) where a hand was 
   detected before and after, indicating MediaPipe briefly lost a visible hand.
   These are interpolated (cubic spline or linear).
2. **Outlier jumps** -- sudden coordinate spikes in detected frames (e.g.
   hand label swaps). These are replaced with interpolated values.

Long gaps where no hand is detected are assumed to be hands leaving the
frame and are **left as NaN** -- this is normal behavior in shuffle videos.

All detected segments also receive light Savitzky-Golay smoothing to reduce jitter.

## Summary

| Metric | Value |
|--------|-------|
| Avg flicker frames filled | 104.4 |
| Avg outlier frames fixed | 168.3 |

## Per-Video Details

| Video | Frames | Flicker Filled | Outliers Fixed |
|-------|--------|---------------|----------------|
| 00001 | 555 | 119 | 138 |
| 00002 | 325 | 22 | 86 |
| 00003 | 837 | 99 | 228 |
| 00004 | 562 | 102 | 191 |
| 00005 | 595 | 57 | 198 |
| 00006 | 549 | 55 | 67 |
| 00007 | 662 | 49 | 211 |
| 00008 | 537 | 108 | 161 |
| 00009 | 896 | 196 | 216 |
| 00010 | 665 | 121 | 216 |
| 00011 | 669 | 219 | 170 |
| 00012 | 415 | 93 | 123 |
| 00013 | 423 | 87 | 134 |
| 00014 | 376 | 32 | 74 |
| 00015 | 494 | 96 | 119 |
| 00016 | 565 | 128 | 155 |
| 00017 | 729 | 89 | 212 |
| 00018 | 651 | 153 | 223 |
| 00019 | 768 | 159 | 276 |

## Parameters Used

- Flicker max gap: 5 frames
- Jump threshold: 3.0x median velocity
- Smoothing window: 7
- Smoothing polyorder: 2
