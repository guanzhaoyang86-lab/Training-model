from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .taxonomy import canonicalize_type_name


DEFAULT_NO_ANOMALY_SUMMARY = "No anomaly is detected."


@dataclass(frozen=True)
class EvidenceItem:
    start: int
    end: int
    type: str
    strength: str
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": int(self.start),
            "end": int(self.end),
            "type": canonicalize_type_name(self.type),
            "strength": str(self.strength),
            "direction": str(self.direction),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceItem":
        return cls(
            start=int(payload["start"]),
            end=int(payload["end"]),
            type=canonicalize_type_name(str(payload["type"])),
            strength=str(payload["strength"]),
            direction=str(payload["direction"]),
        )


@dataclass(frozen=True)
class GroundingResult:
    evidence: list[EvidenceItem]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": [item.to_dict() for item in self.evidence],
            "summary": str(self.summary),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GroundingResult":
        evidence = [EvidenceItem.from_dict(item) for item in payload.get("evidence", [])]
        summary = str(payload.get("summary", DEFAULT_NO_ANOMALY_SUMMARY))
        return cls(evidence=evidence, summary=summary)

    @classmethod
    def no_anomaly(cls) -> "GroundingResult":
        return cls(evidence=[], summary=DEFAULT_NO_ANOMALY_SUMMARY)
