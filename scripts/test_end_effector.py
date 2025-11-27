from lerobot.robots import make_robot_from_config
from lerobot.robots.realman.configuration_realman import RealmanEndEffectorConfig


config = RealmanEndEffectorConfig(
    cameras={},
    # init_type='end_effector',
    init_type='none',
    init_state=[0.1, 0.0, 0.7, 0.0, 0.0, 0.0, 1000],
    pose_units=['m', 'm', 'm', 'degree', 'degree', 'degree', 'm'],
    velocity=10,
)
robot = make_robot_from_config(config)
robot.connect()

print(robot._get_ee_state())

robot.disconnect()