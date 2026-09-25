from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw


# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="HandTalk AI",
    page_icon="🤟",
    layout="wide",
)

st.title("🤟 HandTalk AI")
st.subheader("Real-Time Hand Gesture Number Detector")

st.info(
    "Allow camera access when your browser asks. "
    "This application uses the browser camera instead of cv2.VideoCapture()."
)


# ---------------------------------------------------------
# MediaPipe configuration
# ---------------------------------------------------------

mp_hands = mp.solutions.hands


@st.cache_resource
def get_hand_detector():
    """Create one reusable MediaPipe hand detector."""
    return mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


# ---------------------------------------------------------
# Geometry and finger-counting functions
# ---------------------------------------------------------

Landmark = Tuple[float, float, float]


def point(
    landmarks: List[Landmark],
    index: int,
) -> np.ndarray:
    """Return the x/y coordinates of one landmark."""
    return np.array(
        [landmarks[index][0], landmarks[index][1]],
        dtype=np.float32,
    )


def point_distance(
    landmarks: List[Landmark],
    first_index: int,
    second_index: int,
) -> float:
    """Calculate normalized distance between two landmarks."""
    return float(
        np.linalg.norm(
            point(landmarks, first_index)
            - point(landmarks, second_index)
        )
    )


def finger_is_extended(
    landmarks: List[Landmark],
    fingertip_index: int,
    pip_index: int,
) -> bool:
    """
    Determine whether a finger is raised.

    MediaPipe coordinates increase downward, so an extended fingertip
    is normally above its PIP joint.
    """
    fingertip_y = landmarks[fingertip_index][1]
    pip_y = landmarks[pip_index][1]

    return fingertip_y < pip_y - 0.02


def thumb_is_extended(
    landmarks: List[Landmark],
    handedness: str,
) -> bool:
    """Determine whether the thumb is extended."""
    thumb_tip = point(landmarks, 4)
    thumb_ip = point(landmarks, 3)
    thumb_mcp = point(landmarks, 2)

    hand_width = max(
        point_distance(landmarks, 5, 17),
        0.001,
    )

    thumb_length = float(
        np.linalg.norm(thumb_tip - thumb_mcp)
    )

    # Reject a thumb that is too close to its base.
    if thumb_length < hand_width * 0.45:
        return False

    # The image is mirrored by most browser cameras. MediaPipe's
    # handedness is used to check the thumb direction.
    if handedness == "Right":
        points_outward = thumb_tip[0] < thumb_ip[0]
    else:
        points_outward = thumb_tip[0] > thumb_ip[0]

    return points_outward


def count_fingers(
    landmarks: List[Landmark],
    handedness: str,
) -> int:
    """Count the number of raised fingers on one hand."""
    total = 0

    if thumb_is_extended(landmarks, handedness):
        total += 1

    other_fingers = [
        (8, 6),    # index
        (12, 10),  # middle
        (16, 14),  # ring
        (20, 18),  # pinky
    ]

    for fingertip_index, pip_index in other_fingers:
        if finger_is_extended(
            landmarks,
            fingertip_index,
            pip_index,
        ):
            total += 1

    return total


# ---------------------------------------------------------
# Gesture stability
# ---------------------------------------------------------

class GestureStabilizer:
    """Accept a number only after repeated matching frames."""

    def __init__(self) -> None:
        self.history: deque[int] = deque(maxlen=6)
        self.last_stable_number: Optional[int] = None

    def update(self, number: Optional[int]) -> Optional[int]:
        if number is None:
            self.history.clear()
            return None

        self.history.append(number)

        if len(self.history) < 4:
            return None

        recent_values = list(self.history)[-4:]

        if len(set(recent_values)) != 1:
            return None

        stable_number = recent_values[0]

        if stable_number == self.last_stable_number:
            return None

        self.last_stable_number = stable_number
        return stable_number

    def reset(self) -> None:
        self.history.clear()
        self.last_stable_number = None

    def confidence(self) -> float:
        if not self.history:
            return 0.0

        most_common = max(
            self.history.count(value)
            for value in set(self.history)
        )

        return most_common / len(self.history)


# ---------------------------------------------------------
# Landmark drawing
# ---------------------------------------------------------

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_hand(
    image: Image.Image,
    landmarks: List[Landmark],
) -> None:
    """Draw hand landmarks using Pillow."""
    draw = ImageDraw.Draw(image)
    width, height = image.size

    pixel_points = [
        (
            int(x * width),
            int(y * height),
        )
        for x, y, _ in landmarks
    ]

    for first, second in HAND_CONNECTIONS:
        draw.line(
            [
                pixel_points[first],
                pixel_points[second],
            ],
            fill=(0, 255, 0),
            width=3,
        )

    for x, y in pixel_points:
        radius = 5
        draw.ellipse(
            [
                x - radius,
                y - radius,
                x + radius,
                y + radius,
            ],
            fill=(255, 0, 0),
        )


# ---------------------------------------------------------
# MediaPipe processing
# ---------------------------------------------------------

def process_image(
    image_array: np.ndarray,
) -> Tuple[np.ndarray, List[dict]]:
    """Detect hands, count fingers, and annotate the image."""
    detector = get_hand_detector()

    result = detector.process(image_array)

    annotated = Image.fromarray(image_array.copy())
    detected_hands: List[dict] = []

    if not result.multi_hand_landmarks:
        return np.array(annotated), detected_hands

    handedness_data = result.multi_handedness or []

    for index, hand_landmarks in enumerate(
        result.multi_hand_landmarks
    ):
        landmarks: List[Landmark] = [
            (
                landmark.x,
                landmark.y,
                landmark.z,
            )
            for landmark in hand_landmarks.landmark
        ]

        handedness = "Unknown"
        detection_score = 0.0

        if index < len(handedness_data):
            classification = (
                handedness_data[index].classification[0]
            )
            handedness = classification.label
            detection_score = float(classification.score)

        draw_hand(annotated, landmarks)

        fingers = count_fingers(
            landmarks,
            handedness,
        )

        detected_hands.append(
            {
                "handedness": handedness,
                "fingers": fingers,
                "confidence": detection_score,
            }
        )

    return np.array(annotated), detected_hands


# ---------------------------------------------------------
# Main Streamlit application
# ---------------------------------------------------------

def main() -> None:
    if "stabilizer" not in st.session_state:
        st.session_state.stabilizer = GestureStabilizer()

    if "last_number" not in st.session_state:
        st.session_state.last_number = None

    camera_image = st.camera_input(
        "Show your hand to the camera"
    )

    if camera_image is None:
        st.warning(
            "Waiting for camera access and an image..."
        )
        return

    try:
        pil_image = Image.open(camera_image).convert("RGB")
        image_array = np.array(pil_image)

        annotated_image, hands = process_image(
            image_array
        )

    except Exception as error:
        st.error("The image could not be processed.")
        st.exception(error)
        return

    left_hand = 0
    right_hand = 0
    detection_confidence = 0.0

    for hand in hands:
        if hand["handedness"] == "Left":
            left_hand = hand["fingers"]
        elif hand["handedness"] == "Right":
            right_hand = hand["fingers"]

        detection_confidence += hand["confidence"]

    if hands:
        detection_confidence /= len(hands)
        current_number: Optional[int] = (
            left_hand + right_hand
        )
    else:
        current_number = None
        detection_confidence = 0.0

    stable_number = st.session_state.stabilizer.update(
        current_number
    )

    if stable_number is not None:
        st.session_state.last_number = stable_number

    stability = st.session_state.stabilizer.confidence()

    left_column, right_column = st.columns(
        [2, 1]
    )

    with left_column:
        st.image(
            annotated_image,
            caption="Detected hand landmarks",
            use_container_width=True,
        )

    with right_column:
        st.subheader("Detection result")

        st.metric(
            "Hands detected",
            len(hands),
        )

        st.metric(
            "Left hand",
            left_hand,
        )

        st.metric(
            "Right hand",
            right_hand,
        )

        number_text = st.session_state.last_number

        st.metric(
            "Current number",
            number_text if number_text is not None else "Waiting",
        )

        st.progress(
            min(max(stability, 0.0), 1.0),
            text=f"Gesture stability: {stability:.0%}",
        )

        st.progress(
            min(max(detection_confidence, 0.0), 1.0),
            text=(
                "Estimated detection confidence: "
                f"{detection_confidence:.0%}"
            ),
        )

        if number_text is not None:
            st.success(
                f"You are showing {number_text}."
            )
        else:
            st.info(
                "Place one or two hands clearly in view."
            )

    st.divider()

    if st.button("Reset gesture history"):
        st.session_state.stabilizer.reset()
        st.session_state.last_number = None
        st.rerun()


if __name__ == "__main__":
    main()
