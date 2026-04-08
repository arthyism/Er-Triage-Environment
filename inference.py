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

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# How to connect to the environment
ENV_BASE_URL     = os.getenv("ENV_BASE_URL", "http://localhost:8000")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

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
    """Run one full episode. Returns (success, steps, score, rewards)."""
    rewards:    List[float] = []
    steps_taken = 0
    score        = 0.0
    success      = False
    error_msg    = None

    log_start(task=task_level, env=BENCHMARK, model=MODEL_NAME)

    # Connect to environment — prefer live URL, then Docker, then error
    if LOCAL_IMAGE_NAME:
        env = await ErTriageEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        base_url = ENV_BASE_URL or "http://localhost:8000"
        env = ErTriageEnv(base_url=base_url)

    try:
        result  = await env.reset(task_level=task_level)
        obs     = result.observation
        done    = result.done

        print(f"[DEBUG] Task: {task_level} | Patients: {len(obs.patients)}", flush=True)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Get LLM decision
            response_text = get_model_decision(client, obs.message)
            expected_ids  = [p.patient_id for p in obs.patients]
            decisions     = parse_decisions(response_text, expected_ids)

            # Build action and compact log string
            action = TriageAction(decisions=decisions)
            action_str = json.dumps(
                {"decisions": [
                    {"patient_id": d.patient_id,
                     "esi_level":  d.esi_level,
                     "resource":   d.resource}
                    for d in decisions
                ]},
                separators=(",", ":"),
            )

            # Step environment
            result = await env.step(action)
            reward = result.reward or 0.0
            done   = result.done

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward,
                     done=done, error=error_msg)

            if done:
                break

        # Normalise score to [0, 1]
        score   = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score   = max(1e-6, min(score, 1 - 1e-6))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = str(e)
        log_step(step=steps_taken + 1, action="null", reward=0.0,
                 done=True, error=error_msg)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)

    return success, steps_taken, score, rewards


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

async def main() -> None:
    client     = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    task_levels = ["easy", "medium", "hard"]
    
    cases_per_task = {
        "easy": 7,
        "medium": 4,
        "hard": 3
    }

    avg_scores = []

    for task in task_levels:
        episodes = cases_per_task.get(task, 5)
        print(f"\n[INFO] Starting {task.upper()} task ({episodes} cases)...", flush=True)
        
        task_scores = []
        for i in range(episodes):
            try:
                print(f"[INFO]   -> Running {task.upper()} Case {i+1}/{episodes}", flush=True)
                _, _, score, _ = await run_episode(client, task)
                task_scores.append(score)
            except Exception as e:
                print(f"[ERROR] Task {task} case {i+1} failed: {e}", flush=True)
                task_scores.append(0.0)
                
        task_avg = sum(task_scores) / len(task_scores) if task_scores else 0.0
        avg_scores.append(task_avg)

    avg = sum(avg_scores) / len(avg_scores) if avg_scores else 0.0
    print(
        f"\n[SUMMARY] easy:{avg_scores[0]:.3f} "
        f"medium:{avg_scores[1]:.3f} "
        f"hard:{avg_scores[2]:.3f} "
        f"avg:{avg:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())