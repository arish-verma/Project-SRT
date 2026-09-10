from __future__ import annotations

from datetime import datetime

from app.ai.interfaces import Track
from app.schemas.event import EventRecord, EventSeverity


class RiskEngine:
    """Transparent risk scoring; scores are indicators, not claims of criminal intent."""

    def score(self, track: Track, restricted: bool, night: bool, dwell_seconds: float = 0.0) -> int:
        score = 0
        if restricted and track.label == "person": score += 40
        if night: score += 20
        if dwell_seconds >= 30: score += 15
        if track.label in {"car", "truck", "bus", "motorcycle"}: score += 10
        return min(score, 100)

    @staticmethod
    def severity(score: int) -> EventSeverity:
        if score >= 85: return EventSeverity.CRITICAL
        if score >= 70: return EventSeverity.HIGH
        if score >= 45: return EventSeverity.MEDIUM
        if score >= 20: return EventSeverity.LOW
        return EventSeverity.INFO
