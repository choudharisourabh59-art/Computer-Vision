import os
import random
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from config import (
    CAMERA_INDEX,
    CSV_PATH,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    SPEECH_PHRASES,
    TARGET_FPS,
    WINDOW_NAME,
    DATA_DIR,
    REQUIRED_STABLE_FRAMES,
    COOLDOWN_FRAMES,
    HISTORY_LENGTH
)
from speech.speech_engine import SpeechEngine
from utils.statistics import GestureStatistics
from vision.finger_counter import FingerCounter
from vision.gesture_stabilizer import GestureStabilizer
from vision.hand_detector import HandDetector

# ============================
# Helper functions
# ============================

def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def draw_text_with_background(frame, text, position, font_scale=0.7, color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1):
    x, y = position
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        bg_color,
        thickness + 2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_status_box(frame, label, status, x, y, w, h, color=(30, 200, 90)):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x + 10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, status, (x + 10, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def compute_fps(previous_time):
    current_time = time.time()
    fps = 1 / max((current_time - previous_time), 1e-6)
    return fps, current_time


# ============================
# Main application
# ============================

def main():
    detector = HandDetector(
        max_hands=MAX_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )
    finger_counter = FingerCounter()
    stabilizer = GestureStabilizer(
        required_matches=REQUIRED_STABLE_FRAMES,
        cooldown_frames=COOLDOWN_FRAMES,
        history_size=HISTORY_LENGTH
    )
    speech_engine = SpeechEngine(rate=170, volume=1.0)
    stats = GestureStatistics()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    previous_time = time.time()
    frame_counter = 0

    last_detected_number = None
    last_spoken_text = ""
    debug_mode = False
    current_confidence = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error: Failed to read from camera.")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = frame.copy()
        frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Detect hands
        hand_results = detector.detect(frame_bgr)
        detected_hands = hand_results["hands"]
        left_count = 0
        right_count = 0
        total_count = 0

        # Draw detections and compute counts
        for hand in detected_hands:
            which_hand = hand["handedness"]
            result = finger_counter.count_fingers(hand["landmarks"], which_hand)
            count = result["count"]
            if which_hand == "Left":
                left_count = count
            elif which_hand == "Right":
                right_count = count

            detector.draw(frame_bgr, hand["landmarks_obj"])

        total_count = left_count + right_count

        # Confidence
        confidence = finger_counter.average_confidence(detected_hands)
        current_confidence = confidence

        # Stability logic
        stable_value = stabilizer.update(total_count, confidence)
        if stable_value is not None:
            if last_detected_number != stable_value:
                last_detected_number = stable_value
                phrase_options = SPEECH_PHRASES.get(stable_value, [f"You are showing {stable_value}."])
                speech_text = random.choice(phrase_options)
                speech_engine.speak(speech_text)
                last_spoken_text = speech_text
                stats.register(
                    left_hand=left_count,
                    right_hand=right_count,
                    total=stable_value,
                    confidence=confidence,
                    speech=speech_text
                )
                print(f"Stable number: {stable_value} - {speech_text}")

        # Save result if stable and not duplicate
        if stable_value is not None:
            # Persist to CSV once for that stable gesture
            record = {
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "left_hand": left_count,
                "right_hand": right_count,
                "total": stable_value,
                "confidence": round(confidence, 3),
                "speech": last_spoken_text if last_spoken_text else ""
            }
            try:
                df = pd.read_csv(CSV_PATH)
            except Exception:
                df = pd.DataFrame(columns=["timestamp", "left_hand", "right_hand", "total", "confidence", "speech"])
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
            df.to_csv(CSV_PATH, index=False)

        # Convert back to BGR for display
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)

        # UI layout
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (0, 0), (frame_bgr.shape[1], 80), (18, 18, 30), -1)
        cv2.addWeighted(overlay, 0.9, frame_bgr, 0.1, 0, frame_bgr)

        # Title
        cv2.putText(frame_bgr, "HANDTALK AI", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (79, 226, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, "REAL-TIME GESTURE ASSISTANT", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Side panel
        panel_x = frame_bgr.shape[1] - 310
        panel_y = 90
        cv2.rectangle(frame_bgr, (panel_x, panel_y), (frame_bgr.shape[1] - 20, frame_bgr.shape[0] - 20), (35, 35, 45), -1)
        cv2.rectangle(frame_bgr, (panel_x, panel_y), (frame_bgr.shape[1] - 20, frame_bgr.shape[0] - 20), (80, 180, 255), 2)

        cv2.putText(frame_bgr, "Detected Hands", (panel_x + 20, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"{len(detected_hands)}", (panel_x + 20, panel_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (79, 226, 255), 2, cv2.LINE_AA)

        cv2.putText(frame_bgr, "Left Hand", (panel_x + 20, panel_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, str(left_count), (panel_x + 20, panel_y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame_bgr, "Right Hand", (panel_x + 150, panel_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, str(right_count), (panel_x + 150, panel_y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Current number
        cv2.putText(frame_bgr, "CURRENT NUMBER", (panel_x + 20, panel_y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, str(last_detected_number if last_detected_number is not None else total_count), (panel_x + 20, panel_y + 240), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3, cv2.LINE_AA)

        # Confidence
        cv2.putText(frame_bgr, "CONFIDENCE", (panel_x + 20, panel_y + 290), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"{int(confidence * 100)}%", (panel_x + 20, panel_y + 330), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (79, 226, 255), 2, cv2.LINE_AA)

        # Status
        cv2.putText(frame_bgr, "STATUS", (panel_x + 20, panel_y + 380), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        status_color = (79, 226, 255)
        cv2.putText(frame_bgr, "✓ Hand detected", (panel_x + 20, panel_y + 410), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (89, 245, 154), 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, "✓ Gesture stable", (panel_x + 20, panel_y + 435), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (89, 245, 154), 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, "✓ Speech ready", (panel_x + 20, panel_y + 460), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (89, 245, 154), 1, cv2.LINE_AA)

        # Last spoken
        cv2.putText(frame_bgr, "Last spoken:", (panel_x + 20, panel_y + 500), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        last_text = last_spoken_text if last_spoken_text else "Waiting..."
        cv2.putText(frame_bgr, last_text[:36], (panel_x + 20, panel_y + 530), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # FPS
        fps, previous_time = compute_fps(previous_time)
        cv2.putText(frame_bgr, f"FPS: {int(fps)}", (panel_x + 20, frame_bgr.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Analytics summary in bottom-left area
        analytics_x = 20
        analytics_y = frame_bgr.shape[0] - 90
        cv2.rectangle(frame_bgr, (analytics_x, analytics_y), (analytics_x + 330, analytics_y + 70), (28, 28, 34), -1)
        cv2.rectangle(frame_bgr, (analytics_x, analytics_y), (analytics_x + 330, analytics_y + 70), (120, 120, 140), 1)

        summary = stats.summary()
        cv2.putText(frame_bgr, f"Total gestures: {summary['total_gestures']}", (analytics_x + 12, analytics_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"Unique numbers: {summary['unique_numbers']}", (analytics_x + 12, analytics_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"Top number: {summary['most_common']}", (analytics_x + 12, analytics_y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Debug view
        if debug_mode:
            cv2.putText(frame_bgr, "DEBUG ON", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"Left={left_count} Right={right_count} Total={total_count}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Display frame
        cv2.imshow(WINDOW_NAME, frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            stabilizer.reset()
            last_detected_number = None
            print("Speech history reset.")
        elif key == ord('s'):
            if last_detected_number is not None:
                phrase = random.choice(SPEECH_PHRASES.get(last_detected_number, [f"You are showing {last_detected_number}."]))
                speech_engine.speak(phrase)
                last_spoken_text = phrase
        elif key == ord('d'):
            debug_mode = not debug_mode

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
  
