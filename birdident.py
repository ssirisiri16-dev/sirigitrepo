# accurate_bird_detector.py

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# ===================================
# PAGE CONFIG
# ===================================

st.set_page_config(
    page_title="Bird Detector  App",
    page_icon="🦜",
    layout="centered"
)

# ===================================
# LOAD YOLO MODEL
# ===================================

@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    return model

model = load_model()

# ===================================
# TITLE
# ===================================

st.title("🦜 Accurate Bird Detector ")

st.write(
    "Upload an image and AI will detect whether a bird exists."
)

# ===================================
# FILE UPLOADER
# ===================================

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ===================================
# PROCESS IMAGE
# ===================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Save temp image
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    )

    image.save(temp_file.name)

    # Prediction
    with st.spinner("🔍 Detecting objects..."):

        results = model(temp_file.name)

        detected_objects = []

        for result in results:

            boxes = result.boxes

            for box in boxes:

                class_id = int(box.cls[0])

                class_name = model.names[class_id]

                confidence = float(box.conf[0])

                detected_objects.append(
                    (class_name, confidence)
                )

        # Show results
        st.subheader("Detected Objects")

        bird_found = False

        bird_keywords = [
            "bird",
            "parrot",
            "owl",
            "eagle",
            "duck",
            "hen",
            "peacock",
            "swan"
        ]

        for obj, conf in detected_objects:

            st.write(
                f"✅ {obj} : {conf*100:.2f}%"
            )

            for keyword in bird_keywords:

                if keyword in obj.lower():
                    bird_found = True

        st.markdown("---")

        if bird_found:

            st.success(
                "🦜 Bird Detected Successfully!"
            )

        else:

            st.error(
                "❌ No bird detected."
            )