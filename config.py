# config.py
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "gesture_log.csv"

WINDOW_NAME = "HandTalk AI - Real-Time Gesture Assistant"

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# Hand detection
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Gesture stability
REQUIRED_STABLE_FRAMES = 4
COOLDOWN_FRAMES = 18
HISTORY_LENGTH = 8

# UI
DISPLAY_FONT = cv2 = None  # filled by caller if needed, avoid import here

# Speech settings
DEFAULT_SPEECH_RATE = 170
DEFAULT_VOLUME = 1.0
DEFAULT_VOICE_INDEX = 0

# Phrase bank
SPEECH_PHRASES = {
    0: [
        "I can see zero fingers.",
        "You're showing zero.",
        "Zero fingers detected."
    ],
    1: [
        "I can see one finger.",
        "That's one.",
        "You are showing one."
    ],
    2: [
        "You're showing two.",
        "Nice, that's two fingers.",
        "I can see two."
    ],
    3: [
        "I see three fingers.",
        "That's three.",
        "You are showing three."
    ],
    4: [
        "You're showing four fingers.",
        "I can see four.",
        "That's four."
    ],
    5: [
        "Nice! Five fingers.",
        "You're showing five.",
        "That's five."
    ],
    6: [
        "That's six.",
        "I can see six.",
        "You're showing six."
    ],
    7: [
        "I can see seven.",
        "That's seven.",
        "You are showing seven."
    ],
    8: [
        "You're showing eight.",
        "I can see eight fingers.",
        "That's eight."
    ],
    9: [
        "That's nine.",
        "I can see nine fingers.",
        "You're showing nine."
    ],
    10: [
        "Wow, that's ten fingers.",
        "You are showing ten.",
        "Nice! Ten."
    ]
}

# File system
DATA_DIR.mkdir(exist_ok=True)
if not CSV_PATH.exists():
    CSV_PATH.write_text(
        "timestamp,left_hand,right_hand,total,confidence,speech\n",
        encoding="utf-8"
    )
