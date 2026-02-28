from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

Severity = Literal["P0", "P1", "P2"]
Anomaly = Literal["smoke_fire", "oil_leak", "conveyor_jam", "normal"]

@dataclass
class Signals:
    injury_risk: bool = False
    # leak
    is_spreading: bool = False
    hazard_suspected: bool = False
    # jam
    conveyor_halted: bool = False
    motor_overheating: bool = False
    belt_damage_visible: bool = False

def base_severity(anomaly: Anomaly, s: Signals) -> Severity:
    if anomaly == "smoke_fire":
        return "P0"  # SOP-FIRE-002
    if anomaly == "oil_leak":
        if s.hazard_suspected:
            return "P0"  # flammable/hazardous
        if s.is_spreading:
            return "P1"  # active spreading
        return "P2"      # minor contained
    if anomaly == "conveyor_jam":
        if s.motor_overheating or s.belt_damage_visible:
            return "P0"  # overheating or belt damage
        if s.conveyor_halted:
            return "P1"  # conveyor halted
        return "P2"      # minor cleared quickly
    return "P2"

def apply_global_escalation(
    sev: Severity,
    *,
    injury_risk: bool,
    p1_events_last_30_min: int,
) -> Severity:
    if injury_risk:
        return "P0"  # SOP-SAF-004
    if sev == "P1" and p1_events_last_30_min >= 2:
        return "P0"  # SOP-SAF-004
    return sev

def classify_severity(
    anomaly: Anomaly,
    signals: Signals,
    *,
    p1_events_last_30_min: int = 0,
) -> Severity:
    sev = base_severity(anomaly, signals)
    return apply_global_escalation(
        sev,
        injury_risk=signals.injury_risk,
        p1_events_last_30_min=p1_events_last_30_min,
    )