from dataclasses import dataclass
@dataclass(frozen=True)
class RecoveryDecision: confidence: float; recover: bool
def tracker_confidence(mask_area, previous_area): return min(1., mask_area/max(previous_area,1)) if mask_area else 0.
def recovery_decision(confidence, threshold=.35): return RecoveryDecision(confidence, confidence < threshold)
