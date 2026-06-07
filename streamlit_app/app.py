import streamlit as st
from ultralytics import YOLO
import cv2


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Proctoring System",
    layout="wide"
)

st.title("🚀 AI Proctoring System")

st.write("Real-Time YOLO Object Detection")


# =========================================================
# LOAD YOLO MODEL
# =========================================================

model = YOLO("yolov8n.pt")


# =========================================================
# START CAMERA BUTTON
# =========================================================

run = st.checkbox("Start Camera")


# =========================================================
# VIDEO FRAME PLACEHOLDER
# =========================================================

frame_placeholder = st.empty()


# =========================================================
# OPEN WEBCAM
# =========================================================

camera = cv2.VideoCapture(0)


# =========================================================
# MAIN LOOP
# =========================================================

while run:

    success, frame = camera.read()

    if not success:

        st.error("Failed to access webcam")
        break


    # =====================================================
    # YOLO DETECTION
    # =====================================================

    results = model(frame)

    result = results[0]


    # =====================================================
    # DRAW DETECTIONS
    # =====================================================

    annotated_frame = result.plot()


    # =====================================================
    # ALERT VARIABLES
    # =====================================================

    person_count = 0

    phone_detected = False


    # =====================================================
    # LOOP THROUGH DETECTIONS
    # =====================================================

    for box in result.boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        class_name = model.names[class_id]


        # =================================================
        # PERSON DETECTION
        # =================================================

        if class_name == "person" and confidence > 0.7:

            person_count += 1


        # =================================================
        # PHONE DETECTION
        # =================================================

        if class_name == "cell phone" and confidence > 0.7:

            phone_detected = True


    # =====================================================
    # MULTIPLE PERSON ALERT
    # =====================================================

    if person_count > 1:

        cv2.putText(
            annotated_frame,
            "ALERT: Multiple Persons Detected",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # =====================================================
    # PHONE ALERT
    # =====================================================

    if phone_detected:

        cv2.putText(
            annotated_frame,
            "ALERT: Phone Detected",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # =====================================================
    # DISPLAY FRAME
    # =====================================================

    frame_placeholder.image(
        annotated_frame,
        channels="BGR"
    )


# =========================================================
# RELEASE CAMERA
# =========================================================

camera.release()