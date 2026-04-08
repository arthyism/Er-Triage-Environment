---
title: ER Triage Environment
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - healthcare
  - rl
---

# ER Triage Environment

A sophisticated reinforcement learning environment for training agents to perform clinical triage in emergency departments using the ESI (Emergency Severity Index) protocol.

**Domain:** Emergency Room patient triage and resource allocation  
**Challenge Level:** Medical decision-making under uncertainty  
**Observation:** Patient vitals, history, chief complaints  
**Action:** ESI level assignment + resource allocation + clinical reasoning  

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- OpenAI-compatible API endpoint (HF Inference API, local LLM server, etc.)

### Setup

```bash
# Install dependencies
cd envs/er_triage
pip install -r requirements.txt  # or: uv sync

# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal, run the inference script
cd ../..  # back to project root
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_api_key"

python inference.py
```

### Docker Deployment

```bash
# Build image
docker build -t er-triage:latest -f envs/er_triage/server/Dockerfile .

# Run container
docker run -p 8000:8000 er-triage:latest
```

## Environment Overview

### Task Levels

The environment presents three difficulty levels:

| Level | Patients | Complexity | Time Est. |
|-------|----------|-----------|-----------|
| **Easy** | 1 | Textbook presentations, clear vital abnormalities | ~5 min |
| **Medium** | 3 | Deliberate traps, borderline cases requiring judgment | ~10 min |
| **Hard** | 6-8 | Resource scarcity, deteriorating patients, time pressure | ~20 min |

### ESI Levels

The ESI (Emergency Severity Index) protocol defines five acuity levels:

- **ESI-1 (Immediate):** Life-threatening, requires immediate intervention (resuscitation bay)
- **ESI-2 (Emergency):** High risk of deterioration, needs rapid assessment (monitored bed)
- **ESI-3 (Urgent):** Stable but requires 2+ resources (exam room)
- **ESI-4 (Less Urgent):** Requires 1 resource only (exam room)
- **ESI-5 (Non-Urgent):** Minimal resources needed (waiting room)

### Action Space

**TriageAction**: Submit triage decisions for all patients in the scenario

```python
action = TriageAction(
    decisions=[
        PatientDecision(
            patient_id="P1",
            esi_level=2,           # 1-5
            resource="monitored_bed",  # must match ESI level
            reasoning="Elevated HR + low BP suggests shock"
        ),
        ...
    ]
)
```

Valid resource locations:
- `resuscitation_bay` → ESI-1 only
- `monitored_bed` → ESI-2
- `exam_room` → ESI-3, ESI-4
- `waiting_room` → ESI-5

### Observation Space

**TriageObservation**: Contains patient data and auto-formatted LLM prompt

```python
result = env.reset(task_level="medium")
observation = result.observation

# Available fields:
# - observation.patients : list[Patient]
#   - Each patient has: patient_id, age, sex, chief_complaint, vitals, mental_status, arrival_mode, history
# - observation.available_resources : dict[str, int]
#   - Counts of available beds/bays (can constrain decisions)
# - observation.message : str
#   - Pre-formatted plain-text prompt ready for LLM consumption
```

### Reward Function

Reward is calculated per-patient and averaged across the episode:

```
per_patient_reward = 
    esi_score      × 0.50  (1.0 exact, 0.7 ±1 level, 0.4 ±2 levels)
  + resource_score × 0.30  (1.0 correct, 0.5 acceptable, 0.0 wrong)
  + reasoning_score× 0.20  (fraction of key_signals mentioned)

episode_reward = mean(per_patient) + bonuses/penalties
  + critical_bonus: +0.10 if ESI-1 correctly caught
  - resource_penalty: -0.05 per over-allocated slot
  
clipped to [0.0, 1.0]
```

## API Usage

### Python Client (Async)

```python
from er_triage import ErTriageEnv, TriageAction, PatientDecision

async with ErTriageEnv(base_url="http://localhost:8000").sync() as env:
    # Reset with difficulty level
    result = await env.reset(task_level="medium")
    observation = result.observation
    
    # Access the pre-formatted prompt
    print(observation.message)
    
    # Make decisions (would be LLM output in practice)
    decisions = [
        PatientDecision(
            patient_id="P1",
            esi_level=3,
            resource="exam_room",
            reasoning="Stable vitals, needs imaging"
        )
    ]
    
    # Submit action
    action = TriageAction(decisions=decisions)
    result = await env.step(action)
    
    # Check reward
    print(f"Reward: {result.reward:.2f}")
    print(f"Done: {result.done}")  # Always True (single-step)
```

### Docker Image

```python
from er_triage import ErTriageEnv, TriageAction

env = ErTriageEnv.from_docker_image("er-triage:latest")
try:
    result = env.reset(task_level="easy")
    action = TriageAction(decisions=[...])
    result = env.step(action)
finally:
    env.close()
```

## Inference Script

The `inference.py` script at the project root provides an end-to-end example:

1. Starts three sequential episodes (easy → medium → hard)
2. Uses an LLM to parse patient data and generate decisions
3. Logs results in the hackathon format: `[START]` / `[STEP]` / `[END]`
4. Reports final scores and success metrics

### Running Inference

```bash
# Start server
cd envs/er_triage
uvicorn server.app:app --port 8000 &

# Run inference
cd ../..
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token"

python inference.py
```

### Output Format

```
[START] task=easy env=er_triage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"decisions":[{"patient_id":"P1",...}]} reward=0.85 done=true error=null
[END] success=true steps=1 score=0.85 rewards=0.85

[START] task=medium env=er_triage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"decisions":[...]} reward=0.72 done=true error=null
[END] success=true steps=1 score=0.72 rewards=0.72

[START] task=hard env=er_triage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"decisions":[...]} reward=0.65 done=true error=null
[END] success=true steps=1 score=0.65 rewards=0.65

[SUMMARY] Average score across all tasks: 0.73
```

## Development

### Project Structure

```
envs/er_triage/
├── README.md                    # This file
├── openenv.yaml                 # OpenEnv manifest
├── pyproject.toml               # Dependencies
├── models.py                    # Pydantic models (Action, Observation)
├── client.py                    # ErTriageEnv client class
├── __init__.py                  # Package exports
└── server/
    ├── __init__.py
    ├── app.py                   # FastAPI application
    ├── environment.py           # ERTriageEnvironment logic
    ├── scenarios.py             # Pre-built patient scenarios
    ├── esi_grader.py            # Reward calculation
    └── Dockerfile               # Container definition
```

### Running Tests

```bash
# Test the environment locally (no HTTP server)
cd envs/er_triage
python -m pytest test_client.py -v
```

### Extending Scenarios

To add more patient scenarios, edit `server/scenarios.py`:

```python
# Add to SCENARIOS list
{
    "id": "custom_001",
    "task_level": "medium",
    "patients": [...],
    "ground_truth": {...},
    "key_signals": {...}
}
```

## Evaluation Metrics

**Primary Score**: Average reward across all three task levels (easy, medium, hard)

**Success Criteria** (per task):
- Easy: score ≥ 0.60
- Medium: score ≥ 0.50
- Hard: score ≥ 0.40

**Secondary Metrics**:
- ESI accuracy (% exact matches)
- Resource utilization (% within capacity)
- Reasoning quality (key signals mentioned)

## Performance Notes

- **Single-step episodes**: Each reset+step interaction is one complete episode
- **Concurrency**: Server supports up to 10 concurrent environments
- **Latency**: WebSocket connections for efficient multi-session scenarios
- **Timeout**: Inference script should complete all 3 tasks within 20 minutes on 2vCPU/8GB

## References

- [ESI Triage Protocol](https://www.emergencyseverityindex.org/)
- [OpenEnv Documentation](https://meta-pytorch.github.io/openenv/)
- Clinical decision-making under uncertainty (reinforcement learning perspective)

## License

Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under the BSD-style license.
