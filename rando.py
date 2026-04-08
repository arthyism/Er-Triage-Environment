"""
ER Triage Environment — Inference Script
=========================================
MANDATORY ENV VARS:
    API_BASE_URL      The API endpoint for the LLM.
    MODEL_NAME        The model identifier to use for inference.
    HF_TOKEN          Your Hugging Face / API key.
    LOCAL_IMAGE_NAME  Docker image name (if using from_docker_image)
    ENV_BASE_URL      Running HF Space URL (if using live Space)

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import re
import sys
import textwrap
from typing import List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
API_KEY      = os.getenv("HF_TOKEN")     or os.getenv("API_KEY")

# How to connect to the environment — prefer live Space, fallback to Docker
ENV_BASE_URL     = os.getenv("ENV_BASE_URL")       # e.g. https://your-space.hf.space
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")   # e.g. er-triage-env:latest

BENCHMARK              = "er_triage"
MAX_STEPS              = 1      # ER Triage is single-step per episode
TEMPERATURE            = 0.7
MAX_TOKENS             = 2048
SUCCESS_SCORE_THRESHOLD = 0.3

# For single-step environments max reward = 1.0 per episode
MAX_TOTAL_REWARD = 1.0

# ---------------------------------------------------------------------------
# Environment import
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "envs"))

try:
    from er_triage.client import ErTriageEnv
    from er_triage.models import TriageAction, PatientDecision
except ImportError:
    print("[ERROR] Failed to import er_triage. Ensure envs/ is on the path.", flush=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Logging — strict hackathon format
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    action_str = action[:120] + "..." if len(action) > 120 else action
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM Integration
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an experienced ER triage nurse using the ESI (Emergency Severity Index) protocol.

    Assess each patient and respond with ONLY valid JSON — no markdown, no extra text.

    ESI levels:
      1 = Immediate (life threat)       → resuscitation_bay
      2 = Emergency (high risk)         → monitored_bed
      3 = Urgent (2+ resources needed)  → exam_room
      4 = Less urgent (1 resource)      → exam_room
      5 = Non-urgent (no resources)     → waiting_room

    Response format:
    {
        "decisions": [
            {
                "patient_id": "P001",
                "esi_level": 2,
                "resource": "monitored_bed",
                "reasoning": "brief clinical justification referencing key vitals"
            }
        ]
    }
""").strip()


def get_model_decision(client: OpenAI, observation_message: str) -> str:
    """Call LLM with patient observation. Returns raw response text."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": observation_message},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[DEBUG] LLM request failed: {e}", flush=True)
        return ""


def parse_decisions(response_text: str,
                    expected_ids: List[str]) -> List[PatientDecision]:
    """Parse LLM JSON into PatientDecision list. Falls back to safe defaults."""
    decisions = []
    try:
        # Strip markdown fences if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        data = json.loads(response_text.strip())
        for d in data.get("decisions", []):
            decisions.append(PatientDecision(
                patient_id=d.get("patient_id", ""),
                esi_level=int(d.get("esi_level", 3)),
                resource=d.get("resource", "exam_room"),
                reasoning=d.get("reasoning", ""),
            ))
    except Exception as e:
        print(f"[DEBUG] Parse failed: {e} | raw: {response_text[:300]}", flush=True)

    # Ensure every expected patient has a decision
    submitted = {d.patient_id for d in decisions}
    for pid in expected_ids:
        if pid not in submitted:
            print(f"[WARNING] No decision for {pid}, using default", flush=True)
            decisions.append(PatientDecision(
                patient_id=pid,
                esi_level=3,
                resource="exam_room",
                reasoning="Default: not assessed by model",
            ))

    return decisions


# ---------------------------------------------------------------------------
# Episode Runner
# ---------------------------------------------------------------------------

async def run_episode(
    client: OpenAI,
    task_level: str,
) -> tuple[bool, int, float, List[float]]:
    """Run one full episode via HTTP endpoints."""
    rewards:    List[float] = []
    steps_taken = 0
    score        = 0.0
    success      = False
    error_msg    = None

    log_start(task=task_level, env=BENCHMARK, model=MODEL_NAME)

    base_url = ENV_BASE_URL or "http://localhost:8000"

    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as http:

            # reset
            r = await http.post("/reset", json={"task_level": task_level})
            r.raise_for_status()
            reset_data = r.json()
            obs_data   = reset_data.get("observation", reset_data)
            done       = reset_data.get("done", False)

            # Reconstruct patients list for prompt
            from er_triage.models import Patient, TriageObservation
            patients = [Patient(**p) for p in obs_data.get("patients", [])]
            resources = obs_data.get("available_resources", {})
            obs = TriageObservation(patients=patients,
                                    available_resources=resources)

            print(f"[DEBUG] Task: {task_level} | Patients: {len(patients)}", flush=True)

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                # LLM decision
                response_text = get_model_decision(client, obs.message)
                expected_ids  = [p.patient_id for p in patients]
                decisions     = parse_decisions(response_text, expected_ids)

                action_payload = {
                    "decisions": [
                        {"patient_id": d.patient_id,
                         "esi_level":  d.esi_level,
                         "resource":   d.resource,
                         "reasoning":  d.reasoning}
                        for d in decisions
                    ]
                }
                action_str = json.dumps(
                    {"decisions": [{"patient_id": d.patient_id,
                                    "esi_level": d.esi_level,
                                    "resource": d.resource}
                                   for d in decisions]},
                    separators=(",", ":"),
                )

                # step
                s = await http.post("/step", json=action_payload)
                s.raise_for_status()
                step_data = s.json()

                reward = step_data.get("reward", 0.0) or 0.0
                done   = step_data.get("done", True)

                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=action_str, reward=reward,
                         done=done, error=error_msg)

                if done:
                    break

        score   = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = str(e)
        log_step(step=steps_taken + 1, action="null", reward=0.0,
                 done=True, error=error_msg)

    finally:
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)

    return success, steps_taken, score, rewards


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

async def main() -> None:
    if not API_KEY:
        print("[ERROR] HF_TOKEN or API_KEY not set", flush=True)
        sys.exit(1)

    client      = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    task_levels = ["easy", "medium", "hard"]

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "envs"))
    from er_triage.server.scenarios import SCENARIO_POOLS

    task_scores: dict[str, list[float]] = {t: [] for t in task_levels}

    # Header
    print(flush=True)
    print("╔══════════════════════════════════════════════════════╗", flush=True)
    print("║        ER TRIAGE — OpenEnv RL Environment           ║", flush=True)
    print("║        ESI (Emergency Severity Index) Protocol      ║", flush=True)
    print("╠══════════════════════════════════════════════════════╣", flush=True)
    print(f"║  Model   : {MODEL_NAME[:42]:<42} ║", flush=True)
    print(f"║  Tasks   : easy (10) · medium (8) · hard (5)        ║", flush=True)
    print(f"║  Scoring : ESI accuracy 50% · Resource 30% ·        ║", flush=True)
    print(f"║            Reasoning quality 20%                    ║", flush=True)
    print("╚══════════════════════════════════════════════════════╝", flush=True)

    TASK_DESC = {
        "easy":   "1 patient  · textbook ESI · no ambiguity",
        "medium": "3 patients · simultaneous arrival · traps",
        "hard":   "6-8 patients · resource constraints · deterioration",
    }

    for task in task_levels:
        pool         = SCENARIO_POOLS[task]
        n            = len(pool) if EPISODES_PER_TASK is None else EPISODES_PER_TASK
        scenario_ids = [s["scenario_id"] for s in pool[:n]]

        print(flush=True)
        print(f"┌──────────────────────────────────────────────────────┐", flush=True)
        print(f"│  TASK: {task.upper():<6}  {TASK_DESC[task]:<42} │", flush=True)
        print(f"│  Running {n} scenario(s): {', '.join(scenario_ids):<30} │", flush=True)
        print(f"└──────────────────────────────────────────────────────┘", flush=True)

        for i, sid in enumerate(scenario_ids, 1):
            print(f"\n  ── Scenario {sid} ({i}/{n}) ──────────────────────────", flush=True)
            try:
                _, _, score, _ = await run_episode(client, task)
                task_scores[task].append(score)
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                print(f"  Score: [{bar}] {score:.3f}", flush=True)
            except Exception as e:
                print(f"  [ERROR] {sid} failed: {e}", flush=True)
                task_scores[task].append(0.0)

    # Final summary table
    print(flush=True)
    print("╔══════════════════════════════════════════════════════╗", flush=True)
    print("║                  FINAL RESULTS                      ║", flush=True)
    print("╠══════════╦══════════╦══════════╦══════════╦═════════╣", flush=True)
    print("║  Task    ║   Avg    ║   Best   ║  Worst   ║    N    ║", flush=True)
    print("╠══════════╬══════════╬══════════╬══════════╬═════════╣", flush=True)

    all_avgs = []
    for task in task_levels:
        scores = task_scores[task]
        avg    = sum(scores) / len(scores) if scores else 0.0
        best   = max(scores) if scores else 0.0
        worst  = min(scores) if scores else 0.0
        all_avgs.append(avg)
        print(f"║  {task.upper():<8}║  {avg:.3f}   ║  {best:.3f}   ║  {worst:.3f}   ║   {len(scores):<5}║", flush=True)

    overall = sum(all_avgs) / len(all_avgs) if all_avgs else 0.0
    print("╠══════════╬══════════╬══════════╬══════════╬═════════╣", flush=True)
    print(f"║  OVERALL ║  {overall:.3f}   ║          ║          ║   {sum(len(v) for v in task_scores.values()):<5}║", flush=True)
    print("╚══════════╩══════════╩══════════╩══════════╩═════════╝", flush=True)

    # Score interpretation
    print(flush=True)
    if overall >= 0.85:
        grade = "Excellent — LLM demonstrates strong clinical triage reasoning"
    elif overall >= 0.70:
        grade = "Good — LLM handles most cases correctly with minor errors"
    elif overall >= 0.50:
        grade = "Fair — LLM captures broad patterns but misses nuances"
    else:
        grade = "Needs improvement — significant triage errors detected"
    print(f"  Assessment: {grade}", flush=True)
    print(flush=True)


if __name__ == "__main__":
    asyncio.run(main())