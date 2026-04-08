"""
ESI Grader — reward function for the ER Triage environment.

Reward is fully decomposed into three components per patient:

  1. ESI accuracy     (weight: 0.50) — how close is predicted vs ground truth
  2. Resource match   (weight: 0.30) — does the resource fit the ESI level
  3. Reasoning quality(weight: 0.20) — do key clinical signals appear in reasoning

Final episode reward = mean of per-patient rewards, with a bonus
for correctly identifying the single most critical patient (if any ESI-1 exists).

All rewards are in [0.0, 1.0].
"""

import os
import sys
from dataclasses import dataclass

try:
    from ..models import PatientDecision
except (ImportError, ValueError):
    sys.path.insert(0, os.path.dirname(__file__))
    from models import PatientDecision


# ---------------------------------------------------------------------------
# ESI → valid resource mapping
# Used for resource match scoring
# ---------------------------------------------------------------------------

ESI_RESOURCE_MAP: dict[int, str] = {
    1: "resuscitation_bay",
    2: "monitored_bed",
    3: "exam_room",
    4: "exam_room",
    5: "waiting_room",
}

# One level off is acceptable for ESI 3/4 (both use exam_room)
ACCEPTABLE_RESOURCES: dict[int, set[str]] = {
    1: {"resuscitation_bay"},
    2: {"monitored_bed", "resuscitation_bay"},   # over-triaging resource is ok
    3: {"exam_room", "monitored_bed"},
    4: {"exam_room", "waiting_room"},
    5: {"waiting_room", "exam_room"},
}


# ---------------------------------------------------------------------------
# ESI accuracy scoring
# Penalises harder for under-triage (missing critical patients)
# than over-triage (being overly cautious)
# ---------------------------------------------------------------------------

def _esi_score(predicted: int, true: int) -> float:
    """
    Returns a score in [0.0, 1.0] based on ESI distance.

    Under-triage (predicted > true, e.g. giving ESI 3 to an ESI 1 patient)
    is penalised more heavily than over-triage because in real ER settings
    missing a critical patient is far more dangerous than being overly cautious.

    Distance | Over-triage score | Under-triage score
    ---------|-------------------|-------------------
        0    |       1.00        |       1.00
        1    |       0.70        |       0.40
        2    |       0.30        |       0.10
       3+    |       0.00        |       0.00
    """
    diff = predicted - true  # positive = under-triage, negative = over-triage

    if diff == 0:
        return 1.0
    elif diff == 1:    # under-triage by 1 level
        return 0.40
    elif diff == -1:   # over-triage by 1 level
        return 0.70
    elif diff == 2:
        return 0.10
    elif diff == -2:
        return 0.30
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Resource match scoring
# ---------------------------------------------------------------------------

def _resource_score(resource: str, esi_level: int) -> float:
    """
    Returns 1.0 if resource is appropriate for the ESI level,
    0.5 if it's one step off but not dangerous,
    0.0 if it's completely wrong.
    """
    perfect   = ESI_RESOURCE_MAP.get(esi_level)
    acceptable = ACCEPTABLE_RESOURCES.get(esi_level, set())

    if resource == perfect:
        return 1.0
    elif resource in acceptable:
        return 0.5
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Reasoning quality scoring
# ---------------------------------------------------------------------------

def _reasoning_score(reasoning: str, key_signals: list[str]) -> float:
    """
    Checks how many of the expected key clinical signals appear
    in the agent's reasoning text (case-insensitive).

    Returns fraction of signals mentioned, in [0.0, 1.0].
    Empty reasoning → 0.0. No key signals defined → 0.5 (neutral).
    """
    if not reasoning or not reasoning.strip():
        return 0.0
    if not key_signals:
        return 0.5

    reasoning_lower = reasoning.lower()
    matched = sum(1 for signal in key_signals if signal.lower() in reasoning_lower)
    return round(matched / len(key_signals), 3)


# ---------------------------------------------------------------------------
# Per-patient reward
# ---------------------------------------------------------------------------

@dataclass
class PatientRewardBreakdown:
    patient_id:      str
    predicted_esi:   int
    true_esi:        int
    esi_score:       float   # 0.0 – 1.0
    resource_score:  float   # 0.0 – 1.0
    reasoning_score: float   # 0.0 – 1.0
    total:           float   # weighted sum


def grade_patient(
    decision:    PatientDecision,
    ground_truth: dict,           # {"patient_id": ..., "esi_level": ..., "resource": ...}
    key_signals:  list[str],
) -> PatientRewardBreakdown:
    """
    Grade a single patient decision against ground truth.
    Returns a full breakdown so inference.py can display it clearly.
    """
    pred_esi  = decision.esi_level
    true_esi  = ground_truth["esi_level"]

    esi_score      = _esi_score(pred_esi, true_esi)
    resource_score = _resource_score(decision.resource, pred_esi)
    reasoning_score = _reasoning_score(decision.reasoning, key_signals)

    total = round(
        esi_score      * 0.50 +
        resource_score * 0.30 +
        reasoning_score * 0.20,
        3
    )

    return PatientRewardBreakdown(
        patient_id=decision.patient_id,
        predicted_esi=pred_esi,
        true_esi=true_esi,
        esi_score=esi_score,
        resource_score=resource_score,
        reasoning_score=reasoning_score,
        total=total,
    )


# ---------------------------------------------------------------------------
# Episode reward — grades all patients in one action
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRewardBreakdown:
    per_patient:          list[PatientRewardBreakdown]
    base_score:           float   # mean of per-patient totals
    critical_bonus:       float   # +0.10 if ESI-1 patient correctly identified
    resource_penalty:     float   # -0.05 per over-allocated resource
    final_reward:         float   # clipped to [0.0, 1.0]
    task_level:           str
    scenario_id:          str


def grade_episode(
    decisions:           list[PatientDecision],
    scenario:            dict,
    task_level:          str,
    available_resources: dict[str, int],
) -> EpisodeRewardBreakdown:
    """
    Grade all patient decisions for a complete episode.

    Steps:
      1. Match each decision to its ground truth by patient_id
      2. Grade each patient individually
      3. Compute base score = mean of per-patient scores
      4. Apply critical patient bonus
      5. Apply resource over-allocation penalty
      6. Clip final reward to [0.0, 1.0]
    """
    ground_truth_map = {
        gt["patient_id"]: gt for gt in scenario["ground_truth"]
    }
    decision_map = {d.patient_id: d for d in decisions}

    per_patient_results = []

    # --- Grade patients that were decided ---
    for pid, gt in ground_truth_map.items():
        if pid in decision_map:
            patient_signals = gt.get("key_signals", [])
            breakdown = grade_patient(decision_map[pid], gt, patient_signals)
        else:
            # Patient was not triaged at all — zero score
            breakdown = PatientRewardBreakdown(
                patient_id=pid,
                predicted_esi=-1,
                true_esi=gt["esi_level"],
                esi_score=0.0,
                resource_score=0.0,
                reasoning_score=0.0,
                total=0.0,
            )
        per_patient_results.append(breakdown)

    # --- Base score ---
    base_score = round(
        sum(r.total for r in per_patient_results) / len(per_patient_results), 3
    )

    # --- Critical patient bonus ---
    # +0.10 if every ESI-1 patient in the scenario was correctly identified
    esi1_patients = [gt for gt in scenario["ground_truth"] if gt["esi_level"] == 1]
    critical_bonus = 0.0
    if esi1_patients:
        all_critical_correct = all(
            decision_map.get(gt["patient_id"]) is not None
            and decision_map[gt["patient_id"]].esi_level == 1
            for gt in esi1_patients
        )
        if all_critical_correct:
            critical_bonus = 0.10

    # --- Resource over-allocation penalty ---
    # Check if agent assigned more patients to a resource than slots available
    resource_penalty = 0.0
    resource_usage: dict[str, int] = {}
    for d in decisions:
        resource_usage[d.resource] = resource_usage.get(d.resource, 0) + 1

    for resource, used in resource_usage.items():
        capacity = available_resources.get(resource, 999)
        if used > capacity:
            overflow = used - capacity
            resource_penalty += overflow * 0.05  # -0.05 per over-allocated slot

    # --- Final reward ---
    final = max(0.001, min(0.999, round(
        base_score + critical_bonus - resource_penalty, 3
    )))

    return EpisodeRewardBreakdown(
        per_patient=per_patient_results,
        base_score=base_score,
        critical_bonus=critical_bonus,
        resource_penalty=resource_penalty,
        final_reward=final,
        task_level=task_level,
        scenario_id=scenario["scenario_id"],
    )


# ---------------------------------------------------------------------------
# Pretty printer — used by inference.py for readable output
# ---------------------------------------------------------------------------

def format_reward_report(breakdown: EpisodeRewardBreakdown) -> str:
    lines = [
        "",
        f"{'='*60}",
        f"  EPISODE REWARD REPORT",
        f"  Scenario : {breakdown.scenario_id}  |  Task : {breakdown.task_level.upper()}",
        f"{'='*60}",
        f"  {'Patient':<8} {'Pred':>5} {'True':>5} {'ESI':>6} {'Res':>6} {'Rsn':>6} {'Total':>7}",
        f"  {'-'*52}",
    ]

    for p in breakdown.per_patient:
        pred_str = str(p.predicted_esi) if p.predicted_esi != -1 else "—"
        lines.append(
            f"  {p.patient_id:<8} {pred_str:>5} {p.true_esi:>5} "
            f"{p.esi_score:>6.2f} {p.resource_score:>6.2f} "
            f"{p.reasoning_score:>6.2f} {p.total:>7.3f}"
        )

    lines += [
        f"  {'-'*52}",
        f"  Base score          : {breakdown.base_score:.3f}",
        f"  Critical bonus      : +{breakdown.critical_bonus:.3f}",
        f"  Resource penalty    : -{breakdown.resource_penalty:.3f}",
        f"  {'─'*40}",
        f"  FINAL REWARD        : {breakdown.final_reward:.3f}",
        f"{'='*60}",
        "",
    ]
    return "\n".join(lines)