from __future__ import annotations

from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw


# ------------------------------------------------------------
# Application configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="HandTalk AI",
    page_icon="🤟",
    layout="wide",
)

LOG_FILE = Path("gesture_log.csv")

REQUIRED_STABLE_FRAMES = 4
HISTORY_SIZE = 8


# ------------------------------------------------------------
# Speech phrases
# ------------------------------------------------------------

PHRASES: Dict[int, List[str]] = {
    0: [
        "I can see zero fingers.",
        "You're showing zero.",
    ],
    1: [
        "I can see one finger.",
        "That's one.",
        "You are showing one.",
    ],
    2: [
        "You're showing two.",
        "Nice, that's two fingers.",
    ],
    3: [
        "I see three fingers.",
        "That's three.",
    ],
    4: [
        "You're showing four fingers.",
        "I can see four.",
    ],
    5: [
        "Nice! Five fingers.",
        "You're showing five.",
    ],
    6: [
        "That's six.",
        "I can see six.",
    ],
    7: [
        "I can see seven.",
        "That's seven.",
    ],
    8: [
        "You're showing eight.",
        "I can see eight fingers.",
    ],
    9: [
        "That's nine.",
        "I can see nine fingers.",
    ],
    10: [
        "Wow, that's ten fingers.",
        "You are showing ten.",
    ],
}


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def landmark_xy(
    landmarks: List[Tuple[float, float, float]],
    index: int,
) -> np.ndarray:
    """Return an x/y landmark as a NumPy array."""
    return np.array(
        [landmarks[index][0], landmarks[index][1]],
        dtype=np.float32,
    )


def distance(
    landmarks: List[Tuple[float, float, float]],
    first: int,
    second: int,
) -> float:
    """Calculate normalized 2D distance between two landmarks."""
    return float(np.linalg.norm(landmark_xy(landmarks, first) -
                                landmark_xy(landmarks, second)))


def read_image_as_rgb(uploaded_file) -> np.ndarray:
    """Convert the Streamlit camera image into an RGB NumPy array."""
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def ensure_log_file() -> None:
    """Create the CSV log file if it does not already exist."""
    if not LOG_FILE.exists():
        empty_data = pd.DataFrame(
            columns=[
                "timestamp",
                "left_hand",
                "right_hand",
                "total_fingers",
                "confidence",
                "speech",
            ]
        )
        empty_data.to_csv(LOG_FILE, index=False)


def save_gesture(
    left_hand: int,
    right_hand: int,
    total_fingers: int,
    confidence: float,
    speech: str,
) -> None:
    """Save one stable gesture to the CSV file."""
    ensure_log_file()

    row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "left_hand": left_hand,
                "right_hand": right_hand,
                "total_fingers": total_fingers,
                "confidence": round(confidence, 3),
                "speech": speech,
            }
        ]
    )

    row.to_csv(
        LOG_FILE,
        mode="a",
        header=False,
        index=False,
    )


# ------------------------------------------------------------
# Finger counting
# ------------------------------------------------------------

def is_finger_extended(
    landmarks: List[Tuple[float, float, float]],
    tip_index: int,
    pip_index: int,
    margin: float = 0.02,
) -> bool:
    """
    Determine whether a non-thumb finger is extended.

    MediaPipe image coordinates increase downward, so the fingertip
    is above the PIP joint when the finger is raised.
    """
    tip_y = landmarks[tip_index][1]
    pip_y = landmarks[pip_index][1]

    return tip_y < pip_y - margin


def is_thumb_extended(
    landmarks: List[Tuple[float, float, float]],
    handedness: str,
) -> bool:
    """
    Estimate whether the thumb is extended.

    MediaPipe handedness is used because the thumb extends in opposite
    horizontal directions for the left and right hand.
    """
    thumb_tip = landmark_xy(landmarks, 4)
    thumb_ip = landmark_xy(landmarks, 3)
    thumb_mcp = landmark_xy(landmarks, 2)
    index_mcp = landmark_xy(landmarks, 5)

    hand_width = max(distance(landmarks, 5, 17), 0.001)
    thumb_reach = float(np.linalg.norm(thumb_tip - thumb_mcp))

    # The thumb should be sufficiently far from its base.
    far_enough = thumb_reach > hand_width * 0.45

    if handedness == "Right":
        points_outward = thumb_tip[0] < thumb_ip[0]
    else:
        points_outward = thumb_tip[0] > thumb_ip[0]

    # This secondary relationship helps reject a folded thumb.
    away_from_index = abs(float(thumb_tip[0] - index_mcp[0])) > hand_width * 0.15

    return far_enough and points_outward and away_from_index


def count_fingers(
    landmarks: List[Tuple[float, float, float]],
    handedness: str,
) -> int:
    """Count raised fingers on one hand."""
    count = 0

    if is_thumb_extended(landmarks, handedness):
        count += 1

    # Index, middle, ring, pinky.
    finger_pairs = [
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18),
    ]

    for tip_index, pip_index in finger_pairs:
        if is_finger_extended(landmarks, tip_index, pip_index):
            count += 1

    return count


# ------------------------------------------------------------
# Gesture stability
# ------------------------------------------------------------

class GestureStabilizer:
    """Require the same gesture for several frames before accepting it."""

    def __init__(
        self,
        required_frames: int = REQUIRED_STABLE_FRAMES,
        history_size: int = HISTORY_SIZE,
    ) -> None:
        self.required_frames = required_frames
        self.history: deque[int] = deque(maxlen=history_size)
        self.last_stable: Optional[int] = None

    def update(self, value: Optional[int]) -> Optional[int]:
        """Return a stable value only when enough frames agree."""
        if value is None:
            self.history.clear()
            return None

        self.history.append(value)

        if len(self.history) < self.required_frames:
            return None

        recent = list(self.history)[-self.required_frames:]

        if len(set(recent)) == 1:
            candidate = recent[0]

            if candidate != self.last_stable:
                self.last_stable = candidate
                return candidate

        return None

    def reset(self) -> None:
        """Reset the stability history."""
        self.history.clear()
        self.last_stable = None

    def score(self) -> float:
        """Return an estimated stability score from 0.0 to 1.0."""
        if not self.history:
            return 0.0

        counts = Counter(self.history)
        most_common_count = counts.most_common(1)[0][1]

        return most_common_count / len(self.history)


# ------------------------------------------------------------
# MediaPipe processing
# ------------------------------------------------------------

@st.cache_resource
def create_hand_detector():
    """Create and cache the MediaPipe Hands detector."""
    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def detect_hands(
    rgb_image: np.ndarray,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Detect hands and draw landmarks using Pillow.

    Returns:
        annotated RGB image
        list of detected-hand dictionaries
    """
    hands_detector = create_hand_detector()
    result = hands_detector.process(rgb_image)

    annotated_image = Image.fromarray(rgb_image.copy())
    draw = ImageDraw.Draw(annotated_image)

    detected_hands: List[dict] = []

    if not result.multi_hand_landmarks:
        return np.array(annotated_image), detected_hands

    hand_labels = result.multi_handedness or []

    for hand_index, hand_landmarks in enumerate(result.multi_hand_landmarks):
        points = [
            (landmark.x, landmark.y, landmark.z)
            for landmark in hand_landmarks.landmark
        ]

        handedness = "Unknown"
        detection_confidence = 0.0

        if hand_index < len(hand_labels):
            classification = hand_labels[hand_index].classification[0]
            handedness = classification.label
            detection_confidence = float(classification.score)

        image_height, image_width = rgb_image.shape[:2]

        pixel_points = [
            (
                int(point[0] * image_width),
                int(point[1] * image_height),
            )
            for point in points
        ]

        # MediaPipe hand connections.
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]

        for first, second in connections:
            draw.line(
                [pixel_points[first], pixel_points[second]],
                fill=(0, 255, 0),
                width=3,
            )

        for point in pixel_points:
            radius = 5
            draw.ellipse(
                [
                    point[0] - radius,
                    point[1] - radius,
                    point[0] + radius,
                    point[1] + radius,
                ],
                fill=(255, 0, 0),
            )

        finger_count = count_fingers(points, handedness)

        detected_hands.append(
            {
                "handedness": handedness,
                "landmarks": points,
                "finger_count": finger_count,
                "confidence": detection_confidence,
            }
        )

    return np.array(annotated_image), detected_hands


# ------------------------------------------------------------
# Streamlit user interface
# ------------------------------------------------------------

def main() -> None:
    st.title("🤟 HandTalk AI")
    st.caption("Real-Time Hand Gesture Number Detector")

    st.info(
        "This Streamlit version uses your browser camera. "
        "Allow camera permission when prompted."
    )

    if "stabilizer" not in st.session_state:
        st.session_state.stabilizer = GestureStabilizer()

    if "last_number" not in st.session_state:
        st.session_state.last_number = None

    if "last_speech" not in st.session_state:
        st.session_state.last_speech = ""

    if "gesture_count" not in st.session_state:
        st.session_state.gesture_count = 0

    camera_image = st.camera_input("Show your hand")

    if camera_image is None:
        st.warning("Waiting for a camera image...")
        st.stop()

    try:
        rgb_image = read_image_as_rgb(camera_image)
        annotated_image, detected_hands = detect_hands(rgb_image)
    except Exception as error:
        st.error("The hand detector could not process this image.")
        st.exception(error)
        st.stop()

    left_count = 0
    right_count = 0
    total_confidence = 0.0

    for hand in detected_hands:
        if hand["handedness"] == "Left":
            left_count = hand["finger_count"]
        elif hand["handedness"] == "Right":
            right_count = hand["finger_count"]

        total_confidence += hand["confidence"]

    total_fingers = left_count + right_count

    if detected_hands:
        average_detection_confidence = (
            total_confidence / len(detected_hands)
        )
        current_number: Optional[int] = total_fingers
    else:
        average_detection_confidence = 0.0
        current_number = None

    stable_number = st.session_state.stabilizer.update(current_number)

    if stable_number is not None:
        st.session_state.last_number = stable_number
        st.session_state.gesture_count += 1

        speech_options = PHRASES.get(
            stable_number,
            [f"You are showing {stable_number}."],
        )

        # Streamlit cannot play pyttsx3 audio in the user's browser.
        # This text is displayed as the speech response.
        speech_text = speech_options[
            st.session_state.gesture_count % len(speech_options)
        ]

        st.session_state.last_speech = speech_text

        save_gesture(
            left_hand=left_count,
            right_hand=right_count,
            total_fingers=stable_number,
            confidence=average_detection_confidence,
            speech=speech_text,
        )

    stability_score = st.session_state.stabilizer.score()

    left_column, right_column = st.columns([2, 1])

    with left_column:
        st.image(
            annotated_image,
            caption="Detected hand landmarks",
            use_container_width=True,
        )

    with right_column:
        st.subheader("Detection")

        st.metric("Hands detected", len(detected_hands))
        st.metric("Left hand", left_count)
        st.metric("Right hand", right_count)
        st.metric(
            "Current number",
            (
                st.session_state.last_number
                if st.session_state.last_number is not None
                else "Waiting"
            ),
        )

        st.progress(
            min(max(float(stability_score), 0.0), 1.0),
            text=f"Gesture stability: {stability_score:.0%}",
        )

        st.progress(
            min(max(float(average_detection_confidence), 0.0), 1.0),
            text=(
                "Estimated detection confidence: "
                f"{average_detection_confidence:.0%}"
            ),
        )

        if st.session_state.last_speech:
            st.success(
                f"Assistant response: {st.session_state.last_speech}"
            )

    st.divider()

    if LOG_FILE.exists():
        try:
            log_data = pd.read_csv(LOG_FILE)

            if not log_data.empty:
                st.subheader("Gesture statistics")

                stats_a, stats_b, stats_c = st.columns(3)

                with stats_a:
                    st.metric("Stable gestures", len(log_data))

                with stats_b:
                    st.metric(
                        "Unique numbers",
                        log_data["total_fingers"].nunique(),
                    )

                with stats_c:
                    most_common = (
                        log_data["total_fingers"].mode().iloc[0]
                    )
                    st.metric("Most common number", int(most_common))

                st.dataframe(
                    log_data.tail(10),
                    use_container_width=True,
                )
        except Exception as error:
            st.warning(f"Could not display gesture history: {error}")

    if st.button("Reset gesture history"):
        st.session_state.stabilizer.reset()
        st.session_state.last_number = None
        st.session_state.last_speech = ""
        st.rerun()


if __name__ == "__main__":
    main()
