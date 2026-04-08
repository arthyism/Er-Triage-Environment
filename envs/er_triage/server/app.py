import sys, os, traceback
sys.path.insert(0, os.path.dirname(__file__))
from models import TriageAction, TriageObservation
from environment import ERTriageEnvironment
from openenv.core.env_server import create_fastapi_app

def create_er_triage_environment():
    return ERTriageEnvironment()

try:
    app = create_fastapi_app(
        create_er_triage_environment,
        TriageAction,
        TriageObservation,
        max_concurrent_envs=10,
    )
    print("SUCCESS — /ws registered", flush=True)
except Exception as e:
    traceback.print_exc()
    raise  # crash loudly instead of falling back

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()