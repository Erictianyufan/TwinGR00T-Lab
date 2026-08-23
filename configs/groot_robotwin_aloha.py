"""GR00T N1.7 modality contract for RoboTwin's Aloha AgileX embodiment."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

ACTION_HORIZON = 16

robotwin_aloha_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head", "left_wrist", "right_wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "left_gripper", "right_arm", "right_gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["left_arm", "left_gripper", "right_arm", "right_gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(robotwin_aloha_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
