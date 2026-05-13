# Video Black Screen Decoding Troubleshooting

Use this note when the canvas overlay renders correctly but the `<video>` element stays black or fails to seek/play.

## Symptom

- Bounding boxes, ball markers, and court overlays render on the canvas.
- The underlying video area stays black.
- Seeking to a timestamp may show overlays but still no visible frame.

## Actual Root Cause

In this project, the failure was caused by browser video decode compatibility, not GPU rendering.

- Encodes were effectively ending up in a browser-hostile H.264 color/range/profile combination.
- Some files were limited-range / TV-range and rendered nearly black in browser software decoders.
- Stream delivery also needed proper byte-range support so seeking behaved reliably.

## Permanent Fix

Keep the browser-facing file pipeline strict and deterministic.

- Re-encode uploads to browser-safe MP4 with H.264 + AAC.
- Force full-range output and browser-friendly metadata placement.
- Serve the playable file with byte-range support.

### Backend encoding rules

The conversion path should produce a file that is safe for HTML5 playback:

- H.264 video
- AAC audio
- `yuv420p`
- full color range / PC range
- `+faststart`
- baseline-compatible profile when needed for broad browser fallback

Relevant code lives in:

- `backend/app/services/video_service.py`
- `backend/app/routers/videos.py`

## What to Check First

If the issue returns, inspect these first:

1. `ffprobe` output for the playable file.
2. Whether the file is the `.browser.mp4` version, not the raw upload.
3. Response headers from the stream endpoint, especially `Accept-Ranges: bytes`.
4. Browser console for media errors like decode failure or unsupported format.

## Fast Validation Commands

```bash
ffprobe -v error -show_streams -show_format path/to/video.browser.mp4
```

Look for:

- `pix_fmt=yuv420p`
- H.264 video codec
- AAC audio codec
- browser-friendly file size and duration

If playback is black again, confirm the server is not serving the raw upload path by mistake.

## Frontend Notes

The overlay canvas is independent from media decoding. A working overlay does not prove the `<video>` element is healthy.

Frontend file:

- `frontend/src/components/Video/VideoPlayer.jsx`

Useful checks:

- `onError` on the `<video>` element
- `onLoadedMetadata`
- `onCanPlay`
- `preload="metadata"`

## Search Terms

Use these when debugging again:

- `black screen`
- `decode failed`
- `yuv420p`
- `full range`
- `Accept-Ranges`
- `.browser.mp4`
- `ffprobe`

## Rule of Thumb

This project does not require a GPU for normal browser playback. If the video is black, treat it as an encode/streaming compatibility problem first, not a rendering-performance problem.