import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

st.set_page_config(page_title="YOLO Object Detection", layout="centered")

st.title("🚀 YOLOv8 Object Detection App")

st.write("Upload an image and the model will detect objects.")

# Load better model (more accurate than nano)
model = YOLO("yolov8s.pt")   # s = small (better accuracy)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Run detection with confidence threshold
    results = model(image_np, conf=0.5)

    # Get annotated image
    annotated_image = results[0].plot()

    st.image(annotated_image, caption="Detected Image", use_column_width=True)

    # Show detected objects
    st.subheader("Detected Objects:")
    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            st.write(f"{class_name} - {confidence:.2f}")