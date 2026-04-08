"""Er Triage Environment Client."""

import os
import sys
from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import TriageAction, TriageObservation, TriageState
except (ImportError, ValueError):
    sys.path.insert(0, os.path.dirname(__file__))
    from models import TriageAction, TriageObservation, TriageState


class ErTriageEnv(EnvClient[TriageAction, TriageObservation, TriageState]):
    """
    Client for the ER Triage Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each instance gets its own isolated session.

    Usage (async):
        async with ErTriageEnv(base_url="http://localhost:8000") as env:
            result = await env.reset(task_level="easy")
            result = await env.step(action)
            print(result.reward)

    Usage (sync):
        with ErTriageEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset(task_level="easy")
            result = env.step(action)
    """

    def _step_payload(self, action: TriageAction) -> Dict:
        """Convert TriageAction → JSON payload for WebSocket step message."""
        return {
            "decisions": [
                {
                    "patient_id": d.patient_id,
                    "esi_level":  d.esi_level,
                    "resource":   d.resource,
                    "reasoning":  d.reasoning,
                }
                for d in action.decisions
            ],
        }

    def _parse_result(self, payload: Dict) -> StepResult[TriageObservation]:
        """Parse server response → StepResult[TriageObservation]."""
        obs_data = payload.get("observation", {})

        # Only pass fields that TriageObservation actually defines
        observation = TriageObservation(
            patients=obs_data.get("patients", []),
            available_resources=obs_data.get("available_resources", {}),
            message=obs_data.get("message", ""),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> TriageState:
        """Parse server response → TriageState."""
        return TriageState(
            episode_id=payload.get("episode_id", ""),
            task_level=payload.get("task_level", "easy"),
            scenario_id=payload.get("scenario_id", ""),
            step_count=payload.get("step_count", 0),
            total_patients=payload.get("total_patients", 0),
            triaged_count=payload.get("triaged_count", 0),
            current_score=payload.get("current_score", 0.0),
            is_complete=payload.get("is_complete", False),
        )