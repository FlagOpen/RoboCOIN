# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pprint import pformat

from lerobot.robots import RobotConfig

from .robot import Robot


def make_robot_from_config(config: RobotConfig) -> Robot:
    if config.type == "koch_follower":
        from .koch_follower import KochFollower

        return KochFollower(config)
    elif config.type == "so100_follower":
        from .so100_follower import SO100Follower

        return SO100Follower(config)
    elif config.type == "so100_follower_end_effector":
        from .so100_follower import SO100FollowerEndEffector

        return SO100FollowerEndEffector(config)
    elif config.type == "so101_follower":
        from .so101_follower import SO101Follower

        return SO101Follower(config)
    elif config.type == "lekiwi":
        from .lekiwi import LeKiwi

        return LeKiwi(config)
    elif config.type == "stretch3":
        from .stretch3 import Stretch3Robot

        return Stretch3Robot(config)
    elif config.type == "viperx":
        from .viperx import ViperX

        return ViperX(config)
    elif config.type == "hope_jr_hand":
        from .hope_jr import HopeJrHand

        return HopeJrHand(config)
    elif config.type == "hope_jr_arm":
        from .hope_jr import HopeJrArm

        return HopeJrArm(config)
    elif config.type == "bi_so100_follower":
        from .bi_so100_follower import BiSO100Follower

        return BiSO100Follower(config)
    elif config.type == "mock_robot":
        from tests.mocks.mock_robot import MockRobot

        return MockRobot(config)

    # extension robots
    elif config.type == "dummy":
        from .dummy.dummy_robot import DummyRobot

        return DummyRobot(config)
    elif config.type == "dummy_end_effector":
        from .dummy.dummy_robot_end_effector import DummyRobotEndEffector

        return DummyRobotEndEffector(config)
    elif config.type == "bi_dummy":
        from .bi_dummy.bi_dummy_robot import BiDummyRobot

        return BiDummyRobot(config)
    elif config.type == "bi_dummy_end_effector":
        from .bi_dummy.bi_dummy_robot_end_effector import BiDummyRobotEndEffector

        return BiDummyRobotEndEffector(config)
    elif config.type == "piper":
        from .piper.piper import Piper

        return Piper(config)
    elif config.type == "piper_end_effector":
        from .piper.piper_end_effector import PiperEndEffector

        return PiperEndEffector(config)
    elif config.type == "bi_piper":
        from .bi_piper.bi_piper import BiPiper

        return BiPiper(config)
    elif config.type == "bi_piper_end_effector":
        from .bi_piper.bi_piper_end_effector import BiPiperEndEffector

        return BiPiperEndEffector(config)
    elif config.type == "realman":
        from .realman.realman import Realman

        return Realman(config)
    elif config.type == "realman_end_effector":
        from .realman.realman_end_effector import RealmanEndEffector

        return RealmanEndEffector(config)
    elif config.type == "bi_realman":
        from .bi_realman.bi_realman import BiRealman

        return BiRealman(config)
    elif config.type == "bi_realman_end_effector":
        from .bi_realman.bi_realman_end_effector import BiRealmanEndEffector

        return BiRealmanEndEffector(config)
    elif config.type == "ros_robot":
        from .ros_robot.ros_robot import ROSRobot

        return ROSRobot(config)
    elif config.type == "ros_robot_end_effector":
        from .ros_robot.ros_robot_end_effector import ROSRobotEndEffector

        return ROSRobotEndEffector(config)
    elif config.type == "moveit_robot":
        from .moveit_robot.moveit_robot import MoveitRobot

        return MoveitRobot(config)
    elif config.type == "moveit_robot_end_effector":
        from .moveit_robot.moveit_robot_end_effector import MoveitRobotEndEffector

        return MoveitRobotEndEffector(config)
    else:
        raise ValueError(config.type)


def ensure_safe_goal_position(
    goal_present_pos: dict[str, tuple[float, float]], max_relative_target: float | dict[float]
) -> dict[str, float]:
    """Caps relative action target magnitude for safety."""

    if isinstance(max_relative_target, float):
        diff_cap = dict.fromkeys(goal_present_pos, max_relative_target)
    elif isinstance(max_relative_target, dict):
        if not set(goal_present_pos) == set(max_relative_target):
            raise ValueError("max_relative_target keys must match those of goal_present_pos.")
        diff_cap = max_relative_target
    else:
        raise TypeError(max_relative_target)

    warnings_dict = {}
    safe_goal_positions = {}
    for key, (goal_pos, present_pos) in goal_present_pos.items():
        diff = goal_pos - present_pos
        max_diff = diff_cap[key]
        safe_diff = min(diff, max_diff)
        safe_diff = max(safe_diff, -max_diff)
        safe_goal_pos = present_pos + safe_diff
        safe_goal_positions[key] = safe_goal_pos
        if abs(safe_goal_pos - goal_pos) > 1e-4:
            warnings_dict[key] = {
                "original goal_pos": goal_pos,
                "safe goal_pos": safe_goal_pos,
            }

    if warnings_dict:
        logging.warning(
            "Relative goal position magnitude had to be clamped to be safe.\n"
            f"{pformat(warnings_dict, indent=4)}"
        )

    return safe_goal_positions
