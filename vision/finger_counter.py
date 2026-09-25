# vision/finger_counter.py
from typing import List, Dict, Any

import numpy as np


class FingerCounter:
    def __init__(self):
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_pips = [2, 6, 10, 14, 18]
        self.thumb_base = 2
        self.thumb_tip = 4
        self.thumb_ip = 3

    def _distance(self, a, b):
        return np.linalg.norm(np.array(a)[:2] - np.array(b)[:2])

    def _is_thumb_extended(self, landmarks: List[tuple], handedness: str) -> bool:
        # Thumb movement is handled differently because it can move in two directions.
        # Use the hand orientation to determine whether the thumb points outward.
        wrist = landmarks[0]
        thumb_mcp = landmarks[2]
        thumb_ip = landmarks[3]
        thumb_tip = landmarks[4]
        index_mcp = landmarks[5]

        # Compute a rough direction from wrist to index MCP
        wrist_to_index = np.array(index_mcp[:2]) - np.array(wrist[:2])

        # Right hand: thumb extended if thumb tip is to the right of the thumb IP and beyond MCP.
        if handedness == "Right":
            return (
                thumb_tip[0] > thumb_ip[0] and
                thumb_tip[0] > thumb_mcp[0] and
                thumb_tip[1] < thumb_ip[1] + 0.12
            )
        elif handedness == "Left":
            return (
                thumb_tip[0] < thumb_ip[0] and
                thumb_tip[0] < thumb_mcp[0] and
                thumb_tip[1] < thumb_ip[1] + 0.12
            )
        return False

    def _is_finger_extended(self, landmarks: List[tuple], tip_id: int, pip_id: int) -> bool:
        tip = np.array(landmarks[tip_id][:2])
        pip = np.array(landmarks[pip_id][:2])
        # A finger is usually considered extended if tip is above the PIP joint.
        return float(tip[1]) < float(pip[1]) - 0.03

    def count_fingers(self, landmarks: List[tuple], handedness: str) -> Dict[str, Any]:
        if handedness not in ("Left", "Right"):
            handedness = "Right"

        count = 0

        # Thumb
        if self._is_thumb_extended(landmarks, handedness):
            count += 1

        # Other fingers
        for finger_index in range(1, 5):
            tip_id = self.finger_tips[finger_index]
            pip_id = self.finger_pips[finger_index]
            if self._is_finger_extended(landmarks, tip_id, pip_id):
                count += 1

        return {
            "count": count,
            "handedness": handedness
        }

    def average_confidence(self, detected_hands: List[dict]) -> float:
        if not detected_hands:
            return 0.0
        total = sum(hand.get("confidence", 0.0) for hand in detected_hands)
        return total / len(detected_hands)
