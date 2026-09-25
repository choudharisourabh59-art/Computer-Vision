# utils/statistics.py
from collections import Counter
from typing import List, Dict, Any


class GestureStatistics:
    def __init__(self):
        self.records = []

    def register(self, left_hand, right_hand, total, confidence, speech):
        self.records.append({
            "left_hand": left_hand,
            "right_hand": right_hand,
            "total": total,
            "confidence": confidence,
            "speech": speech
        })

    def summary(self) -> Dict[str, Any]:
        if not self.records:
            return {
                "total_gestures": 0,
                "unique_numbers": 0,
                "most_common": "N/A",
                "average_confidence": 0.0,
                "speech_count": 0
            }

        totals = [record["total"] for record in self.records]
        counter = Counter(totals)
        average_confidence = sum(record["confidence"] for record in self.records) / len(self.records)

        return {
            "total_gestures": len(self.records),
            "unique_numbers": len(counter),
            "most_common": counter.most_common(1)[0][0] if counter else "N/A",
            "average_confidence": round(average_confidence, 3),
            "speech_count": len(self.records)
        }
