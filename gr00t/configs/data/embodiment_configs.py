# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


MODALITY_CONFIGS = {
    ##### Pre-registered pretrain configurations #####
    "oxe_droid_relative_eef_relative_joint": {
        "video": ModalityConfig(
            delta_indices=[-15, 0],
            modality_keys=["exterior_image_1_left", "wrist_image_left"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["eef_9d", "gripper_position", "joint_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(40)),
            modality_keys=["eef_9d", "gripper_position", "joint_position"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                    state_key="eef_9d",
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_position",
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joint_position",
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.language.language_instruction"],
        ),
    },
    ##### Pre-registered posttrain configurations #####
    "unitree_g1_sonic": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "projected_gravity",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(64)),
            modality_keys=[
                "motion_token",
                "left_hand_joints",
                "right_hand_joints",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
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
    },
    "unitree_g1_sonic_refmotion": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "projected_gravity",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(64)),
            modality_keys=[
                "smpl_pose",
                "left_hand_joints",
                "right_hand_joints",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
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
    },
    "unitree_g1_sonic_refmotion_joints_benchmark_smi": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "projected_gravity",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(64)),
            modality_keys=[
                "smpl_joints",
                "left_hand_joints",
                "right_hand_joints",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
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
    },
    "unitree_g1_heft": {
        # heft/scalebfm controller. Unlike sonic, the recorded action is the *tracking policy's
        # reference frame* (root pose + joint targets), not a learned latent token. Layout comes
        # from the collected dataset's meta/modality.json (soma collect --backend heft; see
        # g1_bridge/modality.GENERIC_TRACKING_MODALITY_BLOBS):
        #   state  46 = left_leg6 + right_leg6 + waist3 + left_arm7 + right_arm7
        #               + left_hand7 + right_hand7 + projected_gravity3
        #   action 50 = ref_root_pos3 + ref_root_quat4 + ref_joint_pos29   (action.ref_qpos)
        #             + left_hand_joints7 + right_hand_joints7             (action.hands)
        #
        # This config names groups that only a g1_bridge schema_version >= 5 dataset has, and
        # the lookup it feeds (LeRobotEpisodeLoader._build_dataset_statistics) is a bare
        # `self.modality_meta[modality][joint_key]` — so a mismatch is a KeyError naming the
        # group, not a silent degradation. Two renames got it here:
        #   * until 2026-08-11 the hands were a single 2-D open/close scalar named "hands",
        #     from the ORIGINAL heft exporter (external/heft/sim2real/vla/features.py).
        #   * until 2026-08-17 (schema v5) the 7-per-side groups were left_hand/right_hand.
        #     v5 renamed the ACTION groups to left_hand_joints/right_hand_joints so all five
        #     G1 controllers spell them the same way, and added the measured fingers to the
        #     STATE. A v4 dataset therefore needs the previous spelling; it is not mixable
        #     with a v5 one in any case (soma's action_signature refuses the merge).
        #
        # observation.root_orientation (a wxyz quaternion) is also present in these datasets but
        # deliberately not consumed: projected_gravity carries the same attitude information in
        # the form every other embodiment here uses, and taking both would feed the model one
        # quantity twice.
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                # The MEASURED fingers, added in g1_bridge schema_version 5.
                # Without them a heft policy cannot see its own grasp: it
                # predicts a close and never observes whether the fingers got
                # there, so a slipped or failed grasp is invisible to it.
                "left_hand",
                "right_hand",
                "projected_gravity",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(64)),
            modality_keys=[
                "ref_root_pos",
                "ref_root_quat",
                "ref_joint_pos",
                "left_hand_joints",
                "right_hand_joints",
            ],
            # All five groups are absolute targets (the frame the tracking policy is told to
            # track, and the Dex3 finger targets), not deltas — same convention as sonic's
            # tokens and amo's command stream. ref_root_quat is a raw wxyz quaternion under
            # DEFAULT (min/max) normalization; revisit as rot6d if that proves numerically
            # rough. NOTE ref_root_pos/ref_root_quat are episode-relative as of
            # g1_bridge schema_version 3 — older datasets carry an absolute GMR-world root and
            # must be repaired (soma's scripts/fix_heft_ref_frame.py) before being mixed in.
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
            ] * 5,
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "unitree_g1_amo": {
        # amo (external/g1_backend AmoBackend) controller. Its command space is neither a
        # latent token (sonic) nor a reference frame (heft): the policy emits the same
        # thing the teleoperator does -- arm/hand joint targets plus a torso velocity
        # command -- which the AMO whole-body policy tracks. Layout comes straight from
        # the collected dataset's meta/modality.json (soma collect --backend amo):
        #   state  32 = left_hand7 + right_hand7 + left_arm7 + right_arm7 + rpy3 + height1
        #   action 36 = the same 32 + torso_vx + torso_vy + torso_vyaw + target_yaw
        # NOTE: rpy/height are torso pose, not joints -- at deploy the host must supply
        # them as state.rpy / state.height (soma/sim_eval/client.py passes them through
        # build_observation's extra_state); they are not part of the 43-DoF joint vector.
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_hand",
                "right_hand",
                "left_arm",
                "right_arm",
                "rpy",
                "height",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(64)),
            modality_keys=[
                # g1_bridge schema v5: renamed from left_hand/right_hand, and moved
                # out of the flat `action` column into `action.hands` -- so amo's
                # own action column is now 22 wide, not 36.
                "left_hand_joints",
                "right_hand_joints",
                "left_arm",
                "right_arm",
                "rpy",
                "height",
                "torso_vx",
                "torso_vy",
                "torso_vyaw",
                "target_yaw",
            ],
            # Every group is an absolute target/command in its own units (joint angles,
            # torso RPY, base height, body-frame velocities) -- none of them are deltas,
            # same convention as sonic's tokens and heft's reference frame above.
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
            ] * 10,
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "unitree_g1_almi": {
        # almi (external/g1_backend AlmiBackend) controller -- a DUAL-policy checkpoint whose
        # command is split by body half, which is why its action space looks unlike the other
        # three: a twist drives the lower policy, a retargeted pose reference drives the upper
        # one. Layout comes straight from the collected dataset's meta/modality.json
        # (soma collect --sim --backend almi; see g1_bridge/modality.ALMI_MODALITY_BLOBS):
        #   state  32 = left_leg6 + right_leg6 + waist3 + left_arm7 + right_arm7
        #               + projected_gravity3
        #   action 52 = lin_vel2 + ang_vel_z1 + base_height1                     (action.twist)
        #             + waist_rpy3 + l/r_ee_pos3+3 + l/r_ee_quat4+4 + ref_joint_pos17
        #                                                                   (action.upper_ref)
        #             + left_hand7 + right_hand7                            (action.hands)
        #
        # base_height is a commanded posture in METRES, not a velocity: it is the one twist
        # channel that does not go to zero when the operator disengages. Grouped with the twist
        # because that is how ALMI's lower policy consumes it, not because it is a rate.
        #
        # REDUNDANCY, deliberate and load-bearing: waist_rpy, the EE poses and ref_joint_pos all
        # come from ONE retargeted frame, and the upper policy consumes all three, so all three
        # are recorded rather than making deploy re-derive them (recovering the EE pose from the
        # joint reference alone would need a G1 forward-kinematics model on the host). The cost
        # is that a VLA can emit an internally inconsistent triple; waist_rpy is the safe one
        # (AlmiCommands.validated() re-reads it out of the reference's waist entries at deploy,
        # so the reference wins), the EE/joint pair is not. Watch for it before trusting a
        # trained policy's reach.
        #
        # Quaternions (both EE) are raw wxyz under DEFAULT min/max normalization, matching
        # unitree_g1_heft's ref_root_quat. XYZ_ROT6D would be the better representation but
        # expects position and rotation in one group; these are separate modality.json groups,
        # so switching means changing the recorded schema, not just this config.
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                # The measured fingers, added in g1_bridge schema v5 (see heft above).
                "left_hand",
                "right_hand",
                "projected_gravity",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(64)),
            modality_keys=[
                "lin_vel",
                "ang_vel_z",
                "base_height",
                "waist_rpy",
                "left_ee_pos",
                "right_ee_pos",
                "left_ee_quat",
                "right_ee_quat",
                "ref_joint_pos",
                # Renamed from left_hand/right_hand in g1_bridge schema v5.
                "left_hand_joints",
                "right_hand_joints",
            ],
            # Every group is an absolute command in its own units (body-frame velocities,
            # a height in metres, torso-frame EE poses, joint angles) -- none are deltas,
            # same convention as sonic's tokens, heft's reference frame and amo's command
            # stream above. The EE poses are absolute in the TORSO frame, which is a
            # non-EEF quantity as far as this config is concerned: ActionType.EEF selects
            # the model's end-effector head, which expects a world/base-frame gripper pose.
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
            ] * 11,
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "unitree_g1_full_body_with_waist_height_nav_cmd": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(50)),
            modality_keys=[
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "waist",
                "base_height_command",
                "navigate_command",
            ],
            action_configs=[
                # left_arm
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_arm
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # left_hand
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,  # G1 hand is controlled by binary signals like a gripper
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_hand
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,  # G1 hand is controlled by binary signals like a gripper
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # waist
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # base_height_command
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # navigate_command
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
    },
    "libero_sim": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image", "wrist_image"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "simpler_env_widowx": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image_0"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(8)),
            modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "simpler_env_google": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(8)),
            modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
}


def register_modality_config(
    config: dict, embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
):
    assert embodiment_tag.value not in MODALITY_CONFIGS, (
        f"Embodiment tag {embodiment_tag} already registered"
    )
    MODALITY_CONFIGS[embodiment_tag.value] = config
