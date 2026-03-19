# Project Status

## 2026-03-18
- Fixed Ollama `500` / `ReadTimeout` errors in `analyzer.py` when using heavier VLMs (e.g. `qwen2.5vl:7b`) by:
  - downscaling large incoming frames before base64/JPEG encoding (notably for `--source 1` screen capture)
  - increasing the Ollama HTTP request timeout to avoid client-side premature timeouts
- Fixed `object_detector.py` label generation failing for heavier VLMs (timeouts) by:
  - downscaling object crops before JPEG/base64 encoding
  - increasing the Ollama HTTP request timeout
  - fixing `--censor` argparse to avoid the `type=bool` "False becomes True" pitfall
- Reduced relabel churn when boxes flicker by:
  - relaxing track matching to allow “same object” matching using center-distance + size similarity even when IoU is low
  - stopping the label worker from discarding finished labels just because the track temporarily dropped out of `prev_boxes`

