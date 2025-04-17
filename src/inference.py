import cv2
import numpy as np
from PIL import Image
import torch
import onnxruntime as ort
from ultralytics import YOLO

class PPEDetector:
    def __init__(self, pytorch_model_path="models/model.pt", onnx_model_path="models/model.onnx"):
        self.pytorch_model = YOLO(pytorch_model_path) if pytorch_model_path else None
        self.onnx_session = ort.InferenceSession(onnx_model_path) if onnx_model_path else None
        self.input_name = self.onnx_session.get_inputs()[0].name if self.onnx_session else None
        self.classes = {
            0: "person", 1: "ear", 2: "ear-mufs", 3: "face", 4: "face-guard",
            5: "face-mask", 6: "foot", 7: "tool", 8: "glasses", 9: "gloves",
            10: "helmet", 11: "hands", 12: "head", 13: "medical-suit", 14: "shoes",
            15: "safety-suit", 16: "safety-vest"
        }

    def preprocess(self, img, size=(640, 640)):
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dim
        return img

    def detect(self, img, model_type="pytorch"):
        if isinstance(img, Image.Image):
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        if model_type == "pytorch" and self.pytorch_model:
            results = self.pytorch_model(img)
            return results
        elif model_type == "onnx" and self.onnx_session:
            img = self.preprocess(img)
            outputs = self.onnx_session.run(None, {self.input_name: img})
            return outputs  # [boxes, scores, classes]
        return None