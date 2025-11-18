from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig


@CameraConfig.register_subclass("dummy")
@dataclass
class DummyCameraConfig(CameraConfig):
    """
    Configuration class for the DummyCamera.

    Attributes:
        fps: Frames per second for the dummy camera.
        width: Width of the dummy camera frames.
        height: Height of the dummy camera frames.
    """
    
    pass
