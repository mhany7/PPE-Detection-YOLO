import streamlit as st
import torch
from PIL import Image
import numpy as np
import onnxruntime as ort
import cv2
import yaml
from ultralytics import YOLO

# Paths
MODEL_PT_PATH = "models/model.pt"
MODEL_ONNX_PATH = "models/model.onnx"
YAML_PATH = "data/data.yaml"
IMG_SIZE = (640, 640)  # Consistent with training/validation

# Load YAML for class names
def load_yaml_config(config_file):
    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        return config.get("names", {})
    except Exception as e:
        st.error(f"Error loading YAML: {e}")
        return {}

# Load models
@st.cache_resource
def load_pytorch_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading PyTorch model: {e}")
        return None

@st.cache_resource
def load_onnx_model(model_path):
    try:
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        return session, input_name
    except Exception as e:
        st.error(f"Error loading ONNX model: {e}")
        return None, None

# Preprocess image for inference
def preprocess_image(image, size=IMG_SIZE):
    image = image.convert("RGB")
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Predict with PyTorch
def predict_pytorch(model, image):
    try:
        results = model(image)[0]
        return results
    except Exception as e:
        st.error(f"PyTorch inference error: {e}")
        return None

# Predict with ONNX
def predict_onnx(session, input_name, image, class_names, conf_thres=0.25, iou_thres=0.45):
    try:
        outputs = session.run(None, {input_name: image})
        # YOLO ONNX outputs: [boxes, scores, classes]
        # Ultralytics YOLOv8 ONNX typically outputs [1, num_boxes, 4+num_classes]
        output = outputs[0]  # Shape: [1, num_boxes, 4+17]
        boxes = output[0, :, :4]  # x1, y1, x2, y2
        scores = output[0, :, 4:4+len(class_names)]  # Class probabilities
        conf_scores = np.max(scores, axis=1)  # Max confidence per box
        class_ids = np.argmax(scores, axis=1)  # Predicted class IDs

        # Apply confidence and NMS (simplified)
        mask = conf_scores > conf_thres
        boxes = boxes[mask]
        conf_scores = conf_scores[mask]
        class_ids = class_ids[mask]

        # Simplified NMS (using OpenCV)
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), conf_scores.tolist(), conf_thres, iou_thres
        )
        if isinstance(indices, tuple):
            indices = indices[0]  # Handle tuple output
        summary = []
        for idx in indices:
            label = class_names.get(class_ids[idx], f"Class {class_ids[idx]}")
            summary.append({"Class": label, "Confidence": f"{conf_scores[idx]:.2f}"})
        return summary, boxes[indices], conf_scores[indices], class_ids[indices]
    except Exception as e:
        st.error(f"ONNX inference error: {e}")
        return None, None, None, None

# Draw bounding boxes for ONNX
def draw_onnx_boxes(image, boxes, class_ids, conf_scores, class_names):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for box, cls_id, conf in zip(boxes, class_ids, conf_scores):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names.get(cls_id, f'Class {cls_id}')} ({conf:.2f})"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Get detection summary for PyTorch
def get_detection_summary(results, class_names):
    summary = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = class_names.get(cls_id, f"Class {cls_id}")
        summary.append({"Class": label, "Confidence": f"{conf:.2f}"})
    return summary

# Streamlit interface
st.title("SafeScan: PPE Detection")
st.write("Upload an image to detect PPE (helmets, vests, etc.) using YOLOv8s.")

# Load class names from data.yaml
class_names = load_yaml_config(YAML_PATH)
if class_names:
    st.write(f"Supported Classes: {', '.join(class_names.values())}")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Model selection
    model_choice = st.selectbox("Select Model", ["PyTorch", "ONNX"])

    if model_choice == "PyTorch":
        model = load_pytorch_model(MODEL_PT_PATH)
        if model:
            results = predict_pytorch(model, image)
            if results:
                annotated_image = results.plot()
                st.image(annotated_image, caption="Detection Result", use_column_width=True)
                detection_data = get_detection_summary(results, class_names)
                if detection_data:
                    st.subheader("Detected PPE Items")
                    st.table(detection_data)
                else:
                    st.warning("No PPE items detected.")

    elif model_choice == "ONNX":
        session, input_name = load_onnx_model(MODEL_ONNX_PATH)
        if session:
            processed_image = preprocess_image(image, size=IMG_SIZE)
            summary, boxes, conf_scores, class_ids = predict_onnx(session, input_name, processed_image, class_names)
            if summary:
                st.subheader("Detected PPE Items")
                st.table(summary)
                if boxes is not None:
                    annotated_image = draw_onnx_boxes(image, boxes, class_ids, conf_scores, class_names)
                    st.image(annotated_image, caption="Detection Result", use_column_width=True)
            else:
                st.warning("No PPE items detected.")
