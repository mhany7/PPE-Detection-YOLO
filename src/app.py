import streamlit as st
import torch
from PIL import Image
import numpy as np
import onnxruntime as ort
from torchvision import transforms
import yaml
from ultralytics import YOLO

# the path to the model file
model_path = "model.pt"

model = YOLO("model.pt")  

# Load YAML configuration (optional)
#def load_yaml_config(config_file):
#    with open(config_file, "r") as file:
#        config = yaml.safe_load(file)
#    return config

# Example: Load configuration
#config = load_yaml_config('config.yaml')
#st.write(f"Model Name: {config['model']['name']}")
#st.write(f"Categories: {config['model']['categories']}")

import torch

    
def load_pytorch_model(model_path):
    model = YOLO(model_path)
    return model



# Load ONNX model
def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path)
    input_shape = session.get_inputs()[0].shape  # e.g., [1, 3, 416, 416]
    height, width = input_shape[2], input_shape[3]
    return session, (height, width)

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image, size=(416, 416)):
    preprocess = transforms.Compose([
        transforms.Resize(size),  # Resize to the dynamically detected size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = image.convert("RGB")
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Prediction using PyTorch model
def predict_pytorch(model, image):
    results = model(image)[0]
    return results  # contains boxes, classes, confidence, etc.

# Prediction using ONNX model
def predict_onnx(session, image):
    image = image.numpy()
    inputs = {session.get_inputs()[0].name: image}
    outputs = session.run(None, inputs)
    predicted_class = np.argmax(outputs[0], axis=1)
    return predicted_class[0]

def get_detection_summary(results, class_names):
    summary = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = class_names.get(cls_id, f"Class {cls_id}")
        summary.append({"Class": label, "Confidence": f"{conf:.2f}"})
    return summary

# Streamlit interface
st.title("PPE Detection - Image Classification")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
   
    # Load model (PyTorch or ONNX)
    model_choice = st.selectbox("Select Model", ["PyTorch", "ONNX"])

    if model_choice == "PyTorch":
        model_path = "model.pt"
        
        results = predict_pytorch(model, image)  # use original image
        
        # Show annotated image
        annotated_image = results.plot()
        st.image(annotated_image, caption="Detection Result", use_column_width=True)

        # Show detection summary table
        detection_data = get_detection_summary(results, model.names)
        if detection_data:
            st.subheader("Detected PPE Items")
            st.table(detection_data)
        else:
            st.warning("No PPE items detected.")

    elif model_choice == "ONNX":
        model_path = "model.onnx"
        session, input_size = load_onnx_model(model_path)
        processed_image = preprocess_image(image, size=input_size)
        prediction = predict_onnx(session, processed_image)

        st.write(f"Predicted Class: {prediction}")
        






