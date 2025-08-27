import copy
from typing import Any

from lerobot.errors import DeviceNotConnectedError

from .moveit_robot import MoveitRobot
from .configuration_moveit_robot import MoveitRobotEndEffectorConfig
from ..misc import get_standardization, get_transform, get_visualizer


class MoveitRobotEndEffector(MoveitRobot):
    """
    Example:
        ```python
        config = PiperEndEffectorConfig(port="can1", cameras={"front": {"type": "dummy_camera", "height": 480, "width": 640, "fps": 30}})
        robot = PiperEndEffector(config)
        robot.connect()

        # get observation
        observation = robot.get_observation()

        # send action
        action = {"x": 0.1, "y": 0.2, "z": 0.3, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "gripper": 0.5}
        robot.send_action(action)

        robot.disconnect()
        ```
    """

    config_class = MoveitRobotEndEffectorConfig
    name = "moveit_robot_end_effector"

    def __init__(self, config: MoveitRobotEndEffectorConfig):
        super().__init__(config)

        self._base_state = None
        self._delta_with_previous = config.delta_with_previous

        self.standardization = self.config.standardization_fn \
                               if hasattr(config, 'standardization_fn') else get_standardization('dummy')
        self.transform = get_transform(config.control_mode, config.base_euler)
        self.visualizer = get_visualizer(list(self._cameras_ft.keys()), ['arm'], 
                                         [self.standardization.input_transform(config.init_state)], 
                                         'ee_absolute') \
                          if config.visualize else None
    
    @property
    def action_features(self) -> dict[str, Any]:
        actions = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        if self.config.has_gripper:
            actions.append('gripper')
        return {
            each: float for each in actions
        }
    
    def connect(self):
        super().connect()
        self._base_state = self._get_ee_state()
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        state = self._get_ee_state() if self._delta_with_previous else copy.deepcopy(self._base_state)
        state = self.standardization.input_transform(state)

        action = [action[key] for key in self.action_features.keys()]
        action = self.transform(state, action)
        action = self.standardization.output_transform(action)

        self._set_ee_state(action)

        if self.visualizer:
            state = self.standardization.input_transform(self._get_ee_state())
            observation = self.get_observation()
            images = [observation[cam_key] for cam_key in self._cameras_ft.keys()]
            self.visualizer.add(images, [state])
            self.visualizer.plot()

        return {k: v for k, v in zip(self.action_features.keys(), action)}
