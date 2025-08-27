"""
Example command:

1. Dummy robot & dummy policy:

```python
python src/lerobot/scripts/server/robot_client_openpi.py \
    --robot.type=dummy \
    --robot.control_mode=ee_delta_gripper \
    --robot.cameras="{ front: {type: dummy, width: 640, height: 480, fps: 5} }" \
    --robot.init_ee_state="[0, 0, 0, 0, 1.57, 0, 0]" \
    --robot.base_euler="[0, 1.57, 0]" \
    --robot.id=black 
```
"""

import draccus
import time
import traceback
from dataclasses import dataclass

from openpi_client.websocket_client_policy import WebsocketClientPolicy

import sys
sys.path.append('src/')

from lerobot.cameras.dummy.configuration_dummy import DummyCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots import (
    bi_piper,
    bi_realman,
    dummy,
    piper,
    realman,
)
from lerobot.scripts.server.helpers import get_logger


@dataclass
class OpenPIRobotClientConfig:
    robot: RobotConfig
    host: str = "127.0.0.1"
    port: int = 18000
    frequency: int = 10
    prompt: str = "do something"


class OpenPIRobotClient:
    def __init__(self, config: OpenPIRobotClientConfig):
        self.config = config
        self.logger = get_logger('openpi_robot_client')

        self.policy = WebsocketClientPolicy(config.host, config.port)
        self.logger.info(f'Connected to OpenPI server at {config.host}:{config.port}')

        self.robot = make_robot_from_config(config.robot)
        self.logger.info(f'Initialized robot: {self.robot.name}')
    
    def start(self):
        self.logger.info('Starting robot client...')
        self.robot.connect()
    
    def control_loop(self):
        # signal.signal(signal.SIGINT, quit)                                
        # signal.signal(signal.SIGTERM, quit)

        while True:
            obs = self._prepare_observation(self.robot.get_observation())
            self.logger.info(f'Sent observation: {list(obs.keys())}')
            actions = self.policy.infer(obs)['actions']
            for action in actions:
                action = self._prepare_action(action)
                self.logger.info(f'Received action: {action}')
                self.robot.send_action(action)
            time.sleep(1 / self.config.frequency)

    def stop(self):
        self.logger.info('Stopping robot client...')
        self.robot.disconnect()
    
    def _prepare_observation(self, observation):
        state = []
        for key in self.robot._motors_ft.keys():
            assert key in observation, f"Expected key {key} in observation, but got {observation.keys()}"
            state.append(observation[key])
            observation.pop(key)
        
        observation['observation.state'] = state
        return observation
    
    def _prepare_action(self, action):
        assert len(action) == len(self.robot.action_features), \
            f"Action length {len(action)} does not match expected {len(self.robot.action_features)}: {self.robot.action_features.keys()}"
        return {key: action[i] for i, key in enumerate(self.robot.action_features.keys())}


@draccus.wrap()
def main(cfg: OpenPIRobotClientConfig):
    client = OpenPIRobotClient(cfg)
    client.start()

    try:
        client.control_loop()
    except KeyboardInterrupt:
        client.stop()
    except Exception as e:
        client.logger.error(f'Error in control loop: {e}')
        client.logger.error(traceback.format_exc())
    finally:
        client.stop()


if __name__ == "__main__":
    main()
