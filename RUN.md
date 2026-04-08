# ER Triage — Portable Run Guide

## Prerequisites

- Python 3.10+
- `uv` package manager (recommended) OR `pip`

## Setup (one time)

```bash
cd portable/envs/er_triage

# Option A: using uv (recommended — fast, handles everything)
uv sync

# Option B: using pip in a venv
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
pip install "openenv-core[core]>=0.2.2" openai fastapi uvicorn
```

## Step 1 — Start the environment server

```bash
cd portable/envs/er_triage
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000

# Or if you used pip:
# source .venv/bin/activate && uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Leave this terminal open. You should see:
```
SUCCESS — /ws registered
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Verify: open http://127.0.0.1:8000/health in a browser — should return 200.

## Step 2 — Run inference (new terminal)

```bash
cd portable

export ENV_BASE_URL=http://127.0.0.1:8000
export HF_TOKEN=<your_huggingface_token>
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1

uv run --project envs/er_triage python inference.py

# Or if you used pip:
# source envs/er_triage/.venv/bin/activate && python inference.py
```

IMPORTANT: Use `127.0.0.1` not `localhost` (avoids IPv6 issues on macOS).

## Expected Output

```
[INFO] Starting EASY task...
[START] task=easy env=er_triage model=Qwen/Qwen2.5-72B-Instruct
[DEBUG] Task: easy | Patients: 1
[STEP] step=1 action={...} reward=0.XX done=true error=null
[END] success=true steps=1 score=0.XXX rewards=0.XX
...
[SUMMARY] easy:0.XXX medium:0.XXX hard:0.XXX avg:0.XXX
```

All 3 tasks (easy, medium, hard) run sequentially. Takes ~5-10 min total.

## After Inference — Deploy to HF Spaces

```bash
cd portable/envs/er_triage
openenv push
```

Then run the pre-validation script from the hackathon dashboard and submit.

## File Map

```
portable/
├── inference.py                  ← entry point (run from here)
├── RUN.md                        ← this file
└── envs/
    └── er_triage/
        ├── __init__.py
        ├── models.py             ← Pydantic data models
        ├── client.py             ← WebSocket client
        ├── openenv.yaml          ← HF Space config
        ├── pyproject.toml        ← dependencies
        ├── uv.lock               ← locked deps
        ├── README.md             ← env documentation
        └── server/
            ├── __init__.py
            ├── app.py            ← FastAPI app (uvicorn target)
            ├── environment.py    ← core env logic
            ├── esi_grader.py     ← reward function
            ├── scenarios.py      ← 23 patient scenarios
            ├── requirements.txt
            └── Dockerfile        ← for HF Space deployment
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: openenv` | Run `uv sync` inside `envs/er_triage/` first |
| `Connection refused` on inference | Make sure server is running on port 8000 |
| `reward=0.00` for everything | Check HF_TOKEN is valid and model is accessible |
| IPv6 / `localhost` error | Use `ENV_BASE_URL=http://127.0.0.1:8000` |
