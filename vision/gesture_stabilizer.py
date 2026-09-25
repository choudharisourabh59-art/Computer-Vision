# vision/gesture_stabilizer.py
from collections import deque

class GestureStabilizer:
    def __init__(self, required_matches=4, cooldown_frames=18, history_size=8):
        self.required_matches = required_matches
        self.cooldown_frames = cooldown_frames
        self.history = deque(maxlen=history_size)
        self.cooldown_counter = 0
        self.last_output = None

    def update(self, value, confidence=0.0):
        # cooldown blocks repeated speech
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        self.history.append(value)

        if len(self.history) < self.required_matches:
            return None

        recent = list(self.history)[-self.required_matches:]
        if all(v == recent[0] for v in recent):
            candidate = recent[0]
            if self.cooldown_counter == 0 and candidate != self.last_output:
                self.cooldown_counter = self.cooldown_frames
                self.last_output = candidate
                return candidate

        return None

    def reset(self):
        self.history.clear()
        self.cooldown_counter = 0
        self.last_output = None
