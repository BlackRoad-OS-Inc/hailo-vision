# Hailo Vision

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
