"""
Configuration for Bi-Base Leader Teleoperator
"""

from dataclasses import dataclass

from lerobot.teleoperators import TeleoperatorConfig

from ..base_leader import BaseLeaderConfig, BaseLeaderEndEffectorConfig


@TeleoperatorConfig.register_subclass("bi_base_leader")
@dataclass
class BiBaseLeaderConfig(BaseLeaderConfig):
    """
    Configuration for Bi-Base Leader Teleoperator with joint control
    """

    pass


@TeleoperatorConfig.register_subclass("bi_base_leader_end_effector")
@dataclass
class BiBaseLeaderEndEffectorConfig(BiBaseLeaderConfig, BaseLeaderEndEffectorConfig):
    """
    Configuration for Bi-Base Leader Teleoperator with end effector control
    """
    
    pass