import cv2
import mediapipe as mp
import numpy as np
import streamlit as st


st.set_page_config(
    page_title="HandTalk AI",
    page_icon="🤟",
    layout="centered",
)

st.title("🤟 HandTalk AI")
st.subheader("Real-Time Hand Gesture Detector")


camera_image = st.camera_input("Show your hand to the camera")

if camera_image is None:
    st.info("Allow camera access and show your hand.")
    st.stop()


image_bytes = camera_image.getvalue()
image_array = np.frombuffer(image_bytes, dtype=np.uint8)
frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

if frame is None:
    st.error("Could not read the camera image.")
    st.stop()


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
detected_hands = 0

if results.multi_hand_landmarks:
    detected_hands = len(results.multi_hand_landmarks)

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            output_frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
        )

output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

st.image(output_frame, caption="Hand detection result")

if detected_hands:
    st.success(f"Detected hands: {detected_hands}")
else:
    st.warning("No hand detected. Keep your hand clearly visible.")
