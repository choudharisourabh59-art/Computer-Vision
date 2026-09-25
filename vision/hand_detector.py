# vision/hand_detector.py
from typing import List, Dict, Any

import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    def __init__(self, max_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        detected_hands = []

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = "Unknown"
                confidence = 0.0
                if results.multi_handedness and idx < len(results.multi_handedness):
                    classification = results.multi_handedness[idx].classification[0]
                    handedness = classification.label
                    confidence = classification.score

                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

                # Compute bounding box
                xs = [lm[0] for lm in landmarks]
                ys = [lm[1] for lm in landmarks]
                h, w = frame_bgr.shape[:2]
                x_min = int(min(xs) * w)
                y_min = int(min(ys) * h)
                x_max = int(max(xs) * w)
                y_max = int(max(ys) * h)

                detected_hands.append({
                    "handedness": handedness,
                    "landmarks": landmarks,
                    "landmarks_obj": hand_landmarks,
                    "bbox": (x_min, y_min, x_max, y_max),
                    "confidence": confidence
                })

        return {"hands": detected_hands}

    def draw(self, frame_bgr: np.ndarray, hand_landmarks_obj) -> None:
        self.mp_draw.draw_landmarks(
            frame_bgr,
            hand_landmarks_obj,
            mp.solutions.hands.HAND_CONNECTIONS
        )
