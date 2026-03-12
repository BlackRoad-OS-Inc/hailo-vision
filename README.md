# Hailo Vision

[![CI](https://github.com/blackboxprogramming/hailo-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/blackboxprogramming/hailo-vision/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://python.org)
[![Hailo-8](https://img.shields.io/badge/Hailo--8-26_TOPS-FF6B2B.svg)](https://hailo.ai)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-detection-00D4FF.svg)](https://ultralytics.com)
[![Edge AI](https://img.shields.io/badge/edge-real_time-CC00AA.svg)](https://blackroad.io)



Computer vision inference on Hailo-8 AI accelerators (26 TOPS each). Runs YOLOv5 object detection on Raspberry Pi 5 edge hardware via the Hailo Platform SDK 4.23.0.

## Hardware

- **2x Hailo-8** accelerators (52 TOPS total)
- Octavia (Pi 5) — primary inference node
- Cecilia (Pi 5) — secondary

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Upload UI |
| `/health` | GET | Hailo device status |
| `/detect` | POST | Upload image, get YOLOv5 detections (multipart/form-data) |

## Detection Response

```json
{
  "objects": [{"label": "person", "confidence": 0.95, "bbox": [0.1, 0.2, 0.8, 0.9]}],
  "inference_ms": 12,
  "total_ms": 450,
  "device": "Hailo-8 (26 TOPS)",
  "model": "yolov5s"
}
```

## Run

```bash
pip install -r requirements.txt
python server.py  # http://localhost:8200
```

## Test

```bash
pip install pytest
pytest tests/
```
