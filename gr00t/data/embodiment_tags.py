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

from enum import Enum


"""
Embodiment tags are used to identify the robot embodiment in the data.

Naming convention:
<dataset>_<robot_name>

If using multiple datasets, e.g. sim GR1 and real GR1, we can drop the dataset name and use only the robot name.
"""


class EmbodimentTag(Enum):
    """Embodiment tags supported by the GR00T N1.7 checkpoint.

    Pretrain tags (baked into the base model nvidia/GR00T-N1.7-3B, inference-ready):
    - OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT -> "oxe_droid_relative_eef_relative_joint"
    - XDOF                                  -> "xdof_relative_eef_relative_joint"
    - XDOF_SUBTASK                          -> "xdof_relative_eef_relative_joint_subtask"
    - REAL_G1                               -> "real_g1_relative_eef_relative_joints"
    - REAL_R1_PRO_SHARPA                    -> "real_r1_pro_sharpa_relative_eef"
    - REAL_R1_PRO_SHARPA_HUMAN              -> "real_r1_pro_sharpa_relative_eef_human"
    - REAL_R1_PRO_SHARPA_MAXINSIGHTS        -> "real_r1_pro_sharpa_relative_eef_maxinsights"
    - REAL_R1_PRO_SHARPA_MECKA              -> "real_r1_pro_sharpa_relative_eef_mecka"

    Pre-registered posttrain tags (require finetuned checkpoint):
    - UNITREE_G1           -> "unitree_g1_full_body_with_waist_height_nav_cmd"
    - UNITREE_G1_SONIC     -> "unitree_g1_sonic"
    - UNITREE_G1_HEFT      -> "unitree_g1_heft"
    - UNITREE_G1_AMO       -> "unitree_g1_amo"
    - UNITREE_G1_ALMI      -> "unitree_g1_almi"
    - SIMPLER_ENV_GOOGLE   -> "simpler_env_google"
    - SIMPLER_ENV_WIDOWX   -> "simpler_env_widowx"
    - LIBERO_PANDA         -> "libero_sim"

    Finetuning tag (for custom robots):
    - NEW_EMBODIMENT       -> "new_embodiment"

    Use ``EmbodimentTag.resolve(s)`` to look up a tag by name or value,
    case-insensitively.
    """

    ##### Pretrain embodiment tags (in base model processor_config.json) #####

    OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT = "oxe_droid_relative_eef_relative_joint"
    """
    The Open-X-Embodiment DROID robot with relative EEF and relative joint position actions.
    """

    XDOF = "xdof_relative_eef_relative_joint"
    """
    The generic X-DOF robot with relative EEF and relative joint position actions.
    """

    XDOF_SUBTASK = "xdof_relative_eef_relative_joint_subtask"
    """
    The generic X-DOF robot (subtask variant).
    """

    REAL_G1 = "real_g1_relative_eef_relative_joints"
    """
    Real-world Unitree G1 with relative EEF and relative joint actions.
    """

    REAL_R1_PRO_SHARPA = "real_r1_pro_sharpa_relative_eef"
    """
    Real-world R1 Pro Sharpa with relative EEF actions.
    """

    REAL_R1_PRO_SHARPA_HUMAN = "real_r1_pro_sharpa_relative_eef_human"
    """
    Real-world R1 Pro Sharpa with relative EEF actions (human teleop data).
    """

    REAL_R1_PRO_SHARPA_MAXINSIGHTS = "real_r1_pro_sharpa_relative_eef_maxinsights"
    """
    Real-world R1 Pro Sharpa with relative EEF actions (MaxInsights data, single-cam).
    """

    REAL_R1_PRO_SHARPA_MECKA = "real_r1_pro_sharpa_relative_eef_mecka"
    """
    Real-world R1 Pro Sharpa with relative EEF actions (Mecka data, single-cam).
    """

    ##### Pre-registered posttrain embodiment tags #####

    UNITREE_G1 = "unitree_g1_full_body_with_waist_height_nav_cmd"
    """
    The Unitree G1 robot (sim, full-body with waist height and nav commands).
    """

    UNITREE_G1_SONIC = "unitree_g1_sonic"
    """
    The Unitree G1 robot with SONIC whole-body controller. VLA action space is SONIC latents.
    """

    UNITREE_G1_SONIC_REFMOTION = "unitree_g1_sonic_refmotion"
    """
    The Unitree G1 SONIC controller, but VLA outputs SMPL ref-motion (smpl_pose) instead of latents.
    """

    UNITREE_G1_SONIC_REFMOTION_JOINTS_BENCHMARK_SMI = "unitree_g1_sonic_refmotion_joints_benchmark_smi"
    """
    Unitree G1 SONIC, VLA outputs SMPL ref-motion as smpl_joints (72d) — for benchmark SMI / Tabletop datasets.
    """

    UNITREE_G1_HEFT = "unitree_g1_heft"
    """
    The Unitree G1 robot with the heft (motion_tracking sim2real) whole-body controller. VLA action
    space is the tracking policy's reference frame: root pos (3) + root quat wxyz (4) + 29 joint
    targets, plus a 2-dim hand open/close scalar (soma's ``external/heft`` data collection).
    """

    UNITREE_G1_AMO = "unitree_g1_amo"
    """
    The Unitree G1 robot with the amo whole-body controller. VLA action space is the
    teleoperator's own command stream: arm (7+7) and hand (7+7) joint targets, torso
    rpy (3) + base height (1), and a body-frame torso velocity command
    (vx, vy, vyaw) + target_yaw -- 36 dims, state 32 (the same minus the 4 commands).
    """

    UNITREE_G1_ALMI = "unitree_g1_almi"
    """
    The Unitree G1 robot with the ALMI dual-policy whole-body controller (a lower policy
    for the 12 leg joints, an upper one for the 17 waist+arm joints). The VLA action space
    is ALMI's own split command: a body-frame twist for the legs (lin_vel 2 + ang_vel_z 1
    + base_height 1) and a retargeted pose reference for the upper body (waist_rpy 3,
    torso-frame EE pos 3+3 / quat 4+4, and 17 upper joint targets), plus 7+7 Dex3 finger
    targets -- 52 dims, state 32. NOT the 29-D joint target the two policies produce: that
    is a function of this command plus the checkpoint, and storing it would give a VLA a
    second, redundant action space to shortcut through.
    """

    SIMPLER_ENV_GOOGLE = "simpler_env_google"
    """
    The SimplerEnv Google robot.
    """

    SIMPLER_ENV_WIDOWX = "simpler_env_widowx"
    """
    The SimplerEnv WidowX robot.
    """

    LIBERO_PANDA = "libero_sim"
    """
    The LIBERO Panda robot (used for LIBERO-Goal, LIBERO-Object, LIBERO-Spatial, LIBERO-10).
    """

    # New embodiment during post-training
    NEW_EMBODIMENT = "new_embodiment"
    """
    Any new embodiment.
    """

    XPERIENCE_G1_SONIC = "xperience_g1_sonic"
    """
    Human ego-view mocap (xperience-10m) retargeted to Unitree G1 via GMR, with SONIC
    low-latency motion-token actions. State is the retargeted G1 kinematics
    (dof pos/vel + root pose); action is the 64D SONIC low-latency latent.
    """

    @classmethod
    def resolve(cls, tag: "str | EmbodimentTag") -> "EmbodimentTag":
        """Resolve a string to an EmbodimentTag, case-insensitively.

        Matches by enum **name** first (e.g. ``"xdof"`` -> ``XDOF``), then by
        enum **value** (e.g. ``"xdof_relative_eef_relative_joint"`` -> ``XDOF``).

        Raises:
            ValueError: If *tag* does not match any known embodiment.
        """
        if isinstance(tag, cls):
            return tag
        key = tag.strip()
        key_lower = key.lower()
        # Match by enum name (case-insensitive)
        for member in cls:
            if member.name.lower() == key_lower:
                return member
        # Match by enum value (case-insensitive)
        for member in cls:
            if member.value.lower() == key_lower:
                return member

        def _fmt(tags):
            return "\n".join(f"    {m.name:40s} -> {m.value}" for m in tags)

        msg = (
            f"Unknown embodiment tag: {tag!r}\n\n"
            f"  Base model tags (work with nvidia/GR00T-N1.7-3B):\n"
            f"{_fmt(PRETRAIN_TAGS)}\n\n"
            f"  Posttrain tags (require a finetuned checkpoint):\n"
            f"{_fmt(POSTTRAIN_TAGS)}\n\n"
            f"  Finetuning-only tags (for custom robots):\n"
            f"{_fmt(FINETUNE_ONLY_TAGS)}"
        )
        raise ValueError(msg)

    @classmethod
    def reverse_lookup(cls, value: str) -> "str":
        """Map a tag value string back to its enum name, or return the value as-is."""
        for member in cls:
            if member.value == value:
                return member.name
        return value


# Module-level tag category sets (cannot be Enum class attributes).
PRETRAIN_TAGS: frozenset[EmbodimentTag] = frozenset(
    {
        EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT,
        EmbodimentTag.XDOF,
        EmbodimentTag.XDOF_SUBTASK,
        EmbodimentTag.REAL_G1,
        EmbodimentTag.REAL_R1_PRO_SHARPA,
        EmbodimentTag.REAL_R1_PRO_SHARPA_HUMAN,
        EmbodimentTag.REAL_R1_PRO_SHARPA_MAXINSIGHTS,
        EmbodimentTag.REAL_R1_PRO_SHARPA_MECKA,
    }
)
"""Tags baked into the base model (nvidia/GR00T-N1.7-3B) — usable without finetuning."""

POSTTRAIN_TAGS: frozenset[EmbodimentTag] = frozenset(
    {
        EmbodimentTag.UNITREE_G1,
        EmbodimentTag.UNITREE_G1_SONIC,
        EmbodimentTag.UNITREE_G1_SONIC_REFMOTION_JOINTS_BENCHMARK_SMI,
        EmbodimentTag.UNITREE_G1_SONIC_REFMOTION,
        EmbodimentTag.UNITREE_G1_HEFT,
        EmbodimentTag.UNITREE_G1_AMO,
        EmbodimentTag.UNITREE_G1_ALMI,
        EmbodimentTag.SIMPLER_ENV_GOOGLE,
        EmbodimentTag.SIMPLER_ENV_WIDOWX,
        EmbodimentTag.LIBERO_PANDA,
    }
)
"""Tags that require a finetuned checkpoint."""

FINETUNE_ONLY_TAGS: frozenset[EmbodimentTag] = frozenset(
    {
        EmbodimentTag.NEW_EMBODIMENT,
        EmbodimentTag.XPERIENCE_G1_SONIC,
    }
)
"""Tags for custom robots (finetuning only, not in any shipped checkpoint)."""
