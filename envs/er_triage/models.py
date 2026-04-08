# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Er Triage Environment.

The er_triage environment is a simple test environment that echoes back messages.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator
from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class Vitals(BaseModel):
    bp_systolic:      int   = Field(..., description="Systolic blood pressure (mmHg)")
    bp_diastolic:     int   = Field(..., description="Diastolic blood pressure (mmHg)")
    heart_rate:       int   = Field(..., description="Heart rate (bpm)")
    respiratory_rate: int   = Field(..., description="Respiratory rate (breaths/min)")
    spo2:             float = Field(..., description="Oxygen saturation (%)")
    temperature:      float = Field(..., description="Body temperature (°C)")
    pain_score:       int   = Field(..., ge=0, le=10, description="Pain 0-10")


class Patient(BaseModel):
    patient_id:     str
    age:            int
    sex:            str                   # "M" | "F" | "Other"
    chief_complaint: str
    vitals:         Vitals
    mental_status:  str                   # "alert" | "confused" | "unresponsive"
    arrival_mode:   str                   # "walk-in" | "ambulance" | "police"
    history:        Optional[str] = None  # relevant medical history


class PatientDecision(BaseModel):
    patient_id: str
    esi_level:  int  = Field(..., ge=1, le=5, description="ESI level 1 (most critical) to 5 (least)")
    resource:   str  = Field(
        ...,
        description="One of: resuscitation_bay | monitored_bed | exam_room | waiting_room"
    )
    reasoning:  str  = Field(..., description="Clinical justification for this decision")

    @field_validator("resource")
    @classmethod
    def validate_resource(cls, v: str) -> str:
        valid = {"resuscitation_bay", "monitored_bed", "exam_room", "waiting_room"}
        if v not in valid:
            raise ValueError(f"resource must be one of {valid}, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# Action  (agent → environment)
# ---------------------------------------------------------------------------

class TriageAction(Action):
    """
    The agent submits one decision per patient in the current scenario.
    All patients must be triaged in a single action.
    """
    decisions: list[PatientDecision] = Field(
        ...,
        description="One PatientDecision for every patient in the observation"
    )


# ---------------------------------------------------------------------------
# Observation  (environment → agent)
# ---------------------------------------------------------------------------

def _format_message(
    patients: list[Patient],
    available_resources: dict[str, int]
) -> str:
    """
    Pre-formats the observation into a plain-text prompt
    ready to be sent directly to the LLM.
    """
    lines = [
        "You are an experienced ER triage nurse using the ESI (Emergency Severity Index) protocol.",
        "Assess each patient below and assign an ESI level and resource.",
        "",
        "ESI LEVELS:",
        "  1 - Immediate    : life-threatening, requires immediate intervention",
        "  2 - Emergency    : high risk of deterioration, needs rapid assessment",
        "  3 - Urgent       : stable but needs 2+ resources (labs, imaging, IV, etc.)",
        "  4 - Less urgent  : needs 1 resource only",
        "  5 - Non-urgent   : no resources needed, can wait",
        "",
        "RESOURCES:",
        "  resuscitation_bay  → ESI 1",
        "  monitored_bed      → ESI 2",
        "  exam_room          → ESI 3 or 4",
        "  waiting_room       → ESI 5",
        "",
        "=" * 60,
    ]

    for p in patients:
        v = p.vitals
        lines += [
            f"PATIENT {p.patient_id}",
            f"  Age: {p.age} | Sex: {p.sex} | Arrival: {p.arrival_mode}",
            f"  Chief Complaint: {p.chief_complaint}",
            f"  Vitals:",
            f"    BP {v.bp_systolic}/{v.bp_diastolic} mmHg | HR {v.heart_rate} bpm",
            f"    RR {v.respiratory_rate}/min | SpO2 {v.spo2}% | Temp {v.temperature}°C",
            f"    Pain: {v.pain_score}/10",
            f"  Mental Status: {p.mental_status}",
        ]
        if p.history:
            lines.append(f"  History: {p.history}")
        lines.append("")

    lines += [
        "=" * 60,
        "AVAILABLE RESOURCES:",
    ]
    for resource, count in available_resources.items():
        lines.append(f"  {resource.replace('_', ' ').title()}: {count}")

    lines += [
        "",
        "INSTRUCTIONS:",
        "For EVERY patient provide:",
        "  - esi_level  : integer 1-5",
        "  - resource   : resuscitation_bay | monitored_bed | exam_room | waiting_room",
        "  - reasoning  : brief clinical justification referencing key vitals/symptoms",
        "",
        "Respond as a JSON object with key 'decisions', each item containing:",
        "  patient_id, esi_level, resource, reasoning",
    ]

    return "\n".join(lines)


class TriageObservation(Observation):
    patients:             list[Patient]
    available_resources:  dict[str, int] = Field(
        default={
            "resuscitation_bay": 2,
            "monitored_bed":     3,
            "exam_room":         4,
            "waiting_room":      999,
        }
    )
    message: str = ""   # pre-formatted LLM prompt, auto-built in __init__

    def model_post_init(self, __context) -> None:
        # Auto-generate the message from structured data
        # so inference.py can just do: observation.message
        if not self.message:
            object.__setattr__(
                self,
                "message",
                _format_message(self.patients, self.available_resources)
            )


# ---------------------------------------------------------------------------
# State  (internal tracking — not sent to agent)
# ---------------------------------------------------------------------------

class TriageState(State):
    episode_id:     str
    task_level:     str            # "easy" | "medium" | "hard"
    scenario_id:    str
    step_count:     int   = 0
    total_patients: int   = 0
    triaged_count:  int   = 0
    current_score:  float = 0.0
    is_complete:    bool  = False