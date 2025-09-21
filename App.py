import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Load YOLOv8 model
MODEL_PATH = "../models/yolo_sku110k_model/weights/best.pt"  
model = YOLO(MODEL_PATH)

st.title("🛒 Shelf Monitoring System")
st.write("Upload a shelf image to detect products and identify empty spaces.")

# File uploader
uploaded_file = st.file_uploader("Upload a shelf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Shelf Image", use_container_width=True)

    # Run YOLOv8 inference
    results = model.predict(np.array(image))

    # Draw bounding boxes on image
    annotated_frame = results[0].plot()  # numpy array with detections
    st.image(annotated_frame, caption="Detected Products", use_container_width=True)

    # Extract detection info
    detections = results[0].boxes
    num_products = len(detections)

    st.subheader("📊 Detection Summary")
    st.write(f"**Total detected products:** {num_products}")

    # Example heuristic: if fewer than 10 items detected, raise restock alert
    if num_products < 10:
        st.error("⚠️ Low stock detected! Restocking needed.")
    else:
        st.success("✅ Shelf appears well-stocked.")

    # Optional: class-wise breakdown
    class_counts = {}
    for box in detections:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    st.write("**Class-wise product counts:**")
    st.json(class_counts)


