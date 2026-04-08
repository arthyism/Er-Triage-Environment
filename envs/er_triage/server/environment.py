"""
ER Triage Environment — core logic.

Replaces the echo boilerplate from openenv init.
Wires together scenarios, models, and the grader into
the three methods OpenEnv requires: reset(), step(), state().
"""

import uuid
from openenv.core.env_server import Environment

try:
    from ..models import (
        TriageAction,
        TriageObservation,
        TriageState,
    )
    from .scenarios import (
        sample_scenario,
        get_default_resources,
    )
    from .esi_grader import (
        grade_episode,
        format_reward_report,
        EpisodeRewardBreakdown,
    )
except (ImportError, ValueError):
    # Fallback for standalone/Docker execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from models import (
        TriageAction,
        TriageObservation,
        TriageState,
    )
    from scenarios import (
        sample_scenario,
        get_default_resources,
    )
    from esi_grader import (
        grade_episode,
        format_reward_report,
        EpisodeRewardBreakdown,
    )


class ERTriageEnvironment(Environment):
    # Each instance is isolated with its own state
    SUPPORTS_CONCURRENT_SESSIONS = True

    # ------------------------------------------------------------------
    # Internal state — reset on every new episode
    # ------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        self._episode_id:    str   = ""
        self._task_level:    str   = ""
        self._scenario:      dict  = {}
        self._resources:     dict  = {}
        self._step_count:    int   = 0
        self._current_score: float = 0.0
        self._is_complete:   bool  = False
        self._last_breakdown: EpisodeRewardBreakdown | None = None

    # ------------------------------------------------------------------
    # reset()  — start a new episode
    # ------------------------------------------------------------------

    def reset(self, task_level: str = None) -> TriageObservation:
        """
        Start a fresh episode.

        task_level: "easy" | "medium" | "hard"
                    If None, cycles easy → medium → hard round-robin
                    based on total episodes run (useful for inference.py).
        """
        # Determine task level
        if task_level is None:
            # Default to easy if nothing specified
            task_level = "easy"

        # Sample a scenario — avoid immediate repeat
        prev_id = self._scenario.get("scenario_id") if self._scenario else None
        scenario = sample_scenario(task_level, exclude_id=prev_id)

        # Set episode state
        self._episode_id    = str(uuid.uuid4())[:8]
        self._task_level    = task_level
        self._scenario      = scenario
        self._resources     = get_default_resources(task_level, scenario)
        self._step_count    = 0
        self._current_score = 0.0
        self._is_complete   = False
        self._last_breakdown = None

        return TriageObservation(
            patients=scenario["patients"],
            available_resources=self._resources,
            reward=0.0,
            done=False,
        )

    # ------------------------------------------------------------------
    # step()  — receive agent's triage decisions, score them, return result
    # ------------------------------------------------------------------

    def step(self, action: TriageAction) -> TriageObservation:
        """
        Grade the agent's decisions and return the scored observation.

        In OpenEnv, step() returns an Observation. The reward and done
        flag are attached to the StepResult wrapper by the server layer.
        We store them on the instance so state() can report them, and
        the server layer reads them via the _last_breakdown attribute.

        Single-step environment: one action per episode is sufficient
        to triage all patients. Episode ends after the first step.
        """
        if self._is_complete:
            # Episode already done — return terminal observation
            return self._terminal_observation()

        self._step_count += 1

        # Grade the full action against ground truth
        breakdown = grade_episode(
            decisions=action.decisions,
            scenario=self._scenario,
            task_level=self._task_level,
            available_resources=self._resources,
        )

        self._current_score  = breakdown.final_reward
        self._is_complete    = True          # single-step episodes
        self._last_breakdown = breakdown

        # Print reward report to server logs (visible during testing)
        print(format_reward_report(breakdown))

        # Return a closing observation summarising what happened
        return self._terminal_observation()

    # ------------------------------------------------------------------
    # state()  — episode metadata (not sent to agent)
    # ------------------------------------------------------------------

    @property
    def state(self) -> TriageState:
        return TriageState(
            episode_id=self._episode_id,
            task_level=self._task_level,
            scenario_id=self._scenario.get("scenario_id", ""),
            step_count=self._step_count,
            total_patients=len(self._scenario.get("patients", [])),
            triaged_count=len(self._scenario.get("ground_truth", [])),
            current_score=self._current_score,
            is_complete=self._is_complete,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _terminal_observation(self) -> TriageObservation:
        """
        Returned after the episode ends.
        Message summarises the score so inference.py can log it cleanly.
        """
        if self._last_breakdown:
            msg = (
                f"Episode complete. "
                f"Scenario: {self._scenario.get('scenario_id')} | "
                f"Task: {self._task_level} | "
                f"Final reward: {self._current_score:.3f}"
            )
        else:
            msg = "Episode already complete."

        # reward/done are read by OpenEnv serialization for WebSocket clients
        return TriageObservation(
            patients=self._scenario.get("patients", []),
            available_resources=self._resources,
            message=msg,
            reward=self._current_score,
            done=True,
        )

    # ------------------------------------------------------------------
    # OpenEnv reward property
    # Read by the server layer (env_server.py) to attach reward
    # to the StepResult it sends back to the client.
    # ------------------------------------------------------------------

    @property
    def reward(self) -> float:
        return self._current_score

    @property
    def done(self) -> bool:
        return self._is_complete