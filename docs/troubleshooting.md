# Troubleshooting

## Image Size Mismatch
- **Issue**: Validation used `imgsz=416`, preprocessing used 640x640.
- **Fix**: Set `imgsz=640` in validation.

## ONNX Path Error
- **Issue**: `FileNotFoundError: runs/train/exp/weights/best.onnx`.
- **Fix**: Used correct path (`safescan_yolov8s_opt`).

## Batch Size Consistency
- **Note**: Used batch=16 for training and validation to ensure consistent metrics.