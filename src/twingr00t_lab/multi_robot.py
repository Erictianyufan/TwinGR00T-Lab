"""Policy routing contract for the merged G1 + ALOHA home-service demo."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from twingr00t_lab.home_demo import DemoPhase


class RobotRole(StrEnum):
    UNITREE_G1 = "unitree_g1"
    ALOHA = "aloha_bimanual"


@dataclass(frozen=True)
class PolicyRoute:
    role: RobotRole
    embodiment_tag: str
    endpoint: str
    action_owner: str
    upgrade_path: str | None = None


@dataclass(frozen=True)
class MultiRobotCoordinator:
    """Declare which embodiment owns each phase of the continuous task."""

    g1: PolicyRoute = PolicyRoute(
        role=RobotRole.UNITREE_G1,
        embodiment_tag="real_g1_relative_eef_relative_joints",
        endpoint="tcp://127.0.0.1:5556",
        action_owner="GR00T N1.7 REAL_G1 upper body + stable named-joint gait adapter",
        upgrade_path="UNITREE_G1_SONIC posttrain -> GEAR-SONIC WBC latent actions",
    )
    aloha: PolicyRoute = PolicyRoute(
        role=RobotRole.ALOHA,
        embodiment_tag="new_embodiment",
        endpoint="tcp://127.0.0.1:5555",
        action_owner="GR00T N1.7 action chunk -> RoboTwin ALOHA joints",
        upgrade_path="finetuned ALOHA NEW_EMBODIMENT checkpoint",
    )

    def route(self, phase: DemoPhase) -> PolicyRoute:
        if phase in (DemoPhase.ESTABLISH, DemoPhase.TRANSPORT, DemoPhase.HANDOFF):
            return self.g1
        return self.aloha
