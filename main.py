from pathlib import Path

import streamlit as st

# OpenCV is provided by opencv-python-headless in requirements.txt
try:
    import cv2
except ModuleNotFoundError:
    st.error(
        "OpenCV is not installed. Add 'opencv-python-headless' "
        "to requirements.txt and redeploy the application."
    )
    st.stop()

import mediapipe as mp
import numpy as np


st.set_page_config(
    page_title="HandTalk AI",
    page_icon="🤟",
    layout="centered",
)

st.title("🤟 HandTalk AI")
st.subheader("Real-Time Hand Gesture Number Detector")

st.info(
    "Allow camera access when your browser asks. "
    "Upload a camera frame using the camera control below."
)

camera_image = st.camera_input("Show your hand to the camera")

if camera_image is None:
    st.write("Waiting for a camera image...")
    st.stop()


# Convert uploaded camera image into an OpenCV image
image_bytes = camera_image.getvalue()
image_array = np.frombuffer(image_bytes, dtype=np.uint8)
frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

if frame is None:
    st.error("The camera image could not be read.")
    st.stop()


# MediaPipe expects RGB images
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:

    results = hands.process(rgb_frame)

    output_frame = frame.copy()
    hand_count = 0

    if results.multi_hand_landmarks:
        hand_count = len(results.multi_hand_landmarks)

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                output_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
            )

    output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
    st.image(output_frame_rgb, caption="Hand detection result")

    if hand_count:
        st.success(f"Detected hands: {hand_count}")
    else:
        st.warning("No hand detected. Place your hand clearly in front of the camera.")
