"""
Example command:

1. Dummy robot & dummy policy:

```python
python src/lerobot/scripts/server/robot_client_openpi_anno.py \
    --host="127.0.0.1" \
    --port=18000 \
    --robot.type=bi_dummy_end_effector \
    --robot.cameras="{ observation.images.cam_high: {type: dummy, width: 640, height: 480, fps: 5},observation.images.cam_left_wrist: {type: dummy, width: 640, height: 480, fps: 5},observation.images.cam_right_wrist: {type: dummy, width: 640, height: 480, fps: 5}}" \
    --robot.id=black \
    --robot.delta_with="previous" \
    --robot.pose_units="[m, m, m, radian, radian, radian, m]" \
    --robot.model_pose_units="[m, m, m, radian, radian, radian, m]" \
    --task_config_path="lerobot/scripts/server/task_configs/towel_basket"
```

```python
python src/lerobot/scripts/server/robot_client_openpi_anno.py \
    --host="127.0.0.1" \
    --port=18000 \
    --robot.type=realman \
    --robot.cameras="{ observation.images.cam_high: {type: dummy, width: 640, height: 480, fps: 5},observation.images.cam_left_wrist: {type: dummy, width: 640, height: 480, fps: 5},observation.images.cam_right_wrist: {type: dummy, width: 640, height: 480, fps: 5}}" \
    --robot.init_ee_state="[0, 0, 0, 0, 0, 0, 0]" \
    --robot.base_euler="[0, 0, 0]" \
    --robot.id=black \
```

peach
```python
python src/lerobot/scripts/server/robot_client_openpi_anno.py \
  --host="127.0.0.1"     \
  --port=18000     \
  --task_config_path="lerobot/scripts/server/task_configs/towel_basket.py"    \
  --robot.type=bi_realman_end_effector     \
  --robot.ip_left="169.254.128.18"    \
  --robot.port_left=8080     \
  --robot.ip_right="169.254.128.19"     \
  --robot.port_right=8080     \
  --robot.block=False \
  --robot.cameras="{ observation.images.cam_high: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}, observation.images.cam_left_wrist: {type: opencv, index_or_path: 14, width: 640, height: 480, fps: 30},observation.images.cam_right_wrist: {type: opencv, index_or_path: 20, width: 640, height: 480, fps: 30}}"     \
  --robot.init_type="joint"     \
  --robot.id=black     \
  --robot.delta_with=previous
```

"""

import draccus
import importlib
import numpy as np
import time
import threading
import traceback
from dataclasses import dataclass
from sshkeyboard import listen_keyboard, stop_listening

import sys
sys.path.append('src/')

from lerobot.cameras.dummy.configuration_dummy import DummyCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots import (
    bi_dummy,
    bi_piper,
    bi_realman,
    dummy,
    piper,
    realman,
)

from lerobot.scripts.server.robot_client_openpi import OpenPIRobotClient, OpenPIRobotClientConfig
from lerobot.scripts.server.annotators.operators import make_operators_pipeline


class KeyboardListener:
    def __init__(self):
        self._listener = threading.Thread(target=listen_keyboard, args=(self._on_press,))
        self._listener.daemon = True

        self._quit = False
        self._next_subtask = False
        self._success = None
    
    def listen(self):
        self._listener.start()
    
    def reset(self):
        self._quit = False
        self._next_subtask = False
        self._success = None
    
    def _on_press(self, key):
        if key == 's':
            self._next_subtask = True
        
        elif key == 'q':
            self._quit = True
        
        elif key == 'y' and self._quit:
            self._success = True
            stop_listening()
        
        elif key == 'n' and self._quit:
            self._success = False
            stop_listening()
    

@dataclass
class OpenPIWithAnnotationRobotClientConfig(OpenPIRobotClientConfig):
    task_config_path: str = ""


class OpenPIWithAnnotationRobotClient(OpenPIRobotClient):
    def __init__(self, config: OpenPIWithAnnotationRobotClientConfig):
        super().__init__(config)
        self.keyboard_listener = KeyboardListener()
        self._load_task_config(config.task_config_path)

        self._states = []
        self._subtask_index = 0
    
    def start(self):
        super().start()
        for _ in range(10):
            left_state = self.robot.left_robot.model_pose_transform.output_transform(self.robot.left_robot.get_ee_state())
            right_state = self.robot.right_robot.model_pose_transform.output_transform(self.robot.right_robot.get_ee_state())
            state = np.concatenate([left_state, right_state])
            self._states.append({'observation.state': state})

    def _load_task_config(self, path):
        module_name = path.replace('/', '.').rstrip('.py')
        task_module = importlib.import_module(module_name)
        # class name: TaskConfig
        task_config = task_module.TaskConfig()
        self._scene = task_config.scene
        self._task = task_config.task
        self._subtasks = task_config.subtasks
        self._pipeline = make_operators_pipeline(task_config.operaters)
    
    def _prepare_observation(self, observation):
        observation = super()._prepare_observation(observation)
        observation['prompt'] = self._parse_annotation()
        return observation
    
    def _parse_annotation(self):
        annotation = self._pipeline(self._states)

        prompt = 'scene: {}<full_stop>task: {}<comma>{}<full_stop>movement: {} {}.'.format(
            self._scene.split(',')[0],
            self._task,
            self._subtasks[self._subtask_index],
            annotation[-1]['movement_summary_left'],
            annotation[-1]['movement_summary_right'],
        ).lower()
        prompt = (
            prompt.replace('. ', '')
            .replace('.', '')
            .replace('a ', '')
            .replace('the ', '')
            .replace('is ', '')
            .replace('are ', '')
            .replace('<comma>', ', ')
            .replace('<full_stop>', '. ')
        ).strip()

        while '  ' in prompt:
            prompt = prompt.replace('  ', ' ')
        return prompt
    
    def _after_action(self):
        left_state = self.robot.left_robot.model_pose_transform.output_transform(self.robot.left_robot.get_ee_state())
        right_state = self.robot.right_robot.model_pose_transform.output_transform(self.robot.right_robot.get_ee_state())
        state = np.concatenate([left_state, right_state])
        self._states.append({'observation.state': state})

        obs = self.robot.get_observation()
        frames = [obs[key] for key in self.config.camera_keys]
        self.video_recorder.add(frames)

        if self.keyboard_listener._next_subtask:
            self._subtask_index = min(self._subtask_index + 1, len(self._subtasks) - 1)
            self.keyboard_listener.reset()
        
        if self.keyboard_listener._quit:
            print('Success? (y/n): ', end='', flush=True)
            while self.keyboard_listener._success is None:
                time.sleep(0.1)
            print('Got:', self.keyboard_listener._success)
            self.video_recorder.save(task=self._task, success=self.keyboard_listener._success)
            self._is_finished = True


@draccus.wrap()
def main(cfg: OpenPIWithAnnotationRobotClientConfig):
    client = OpenPIWithAnnotationRobotClient(cfg)
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
