"""
This module provides visualization tools for robot trajectories and actions.
It includes classes for recording trajectories and visualizing them alongside images.
It supports various transformations of end effector states, such as absolute and delta representations.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from .transforms import get_transform


class TrajectoryRecorder:
    """
    A class to record and transform end effector trajectories.
    It supports various transformations such as absolute, delta base, and delta gripper.

    Attributes:
        init_state: Initial state of the end effector.
        transform_type: Type of transformation to apply to the actions.
    
    Examples:
        ```python
        recorder = TrajectoryRecorder(init_state=np.array([0, 0, 0, 0, 0, 0, 0]), transform_type='ee_absolute')
        recorder.add(np.array([0.1, 0.1, 0.1, 0, 0, 0, 1]))
        recorder.add(np.array([0.2, 0.2, 0.2, 0, 0, 0, 1]))
        ```
    """

    def __init__(self, init_state, transform_type):
        self.init_state = init_state
        self.transform_type = transform_type

        self.trajectory = {
            'timesteps': [0],
            'positions': [init_state[:3]],
            'euler_angles': [init_state[3:6]],
            'grips': [init_state[6]],
        }
        self.timestep = 0

        self.transform = get_transform(transform_type)

    def add(self, action):
        current_pos, current_euler, current_grip = (
            self.trajectory['positions'][-1], 
            self.trajectory['euler_angles'][-1], 
            self.trajectory['grips'][-1],
        )
        current_state = np.concatenate((current_pos, current_euler, np.array([current_grip])), axis=0)
        transformed_action = self.transform(current_state, action)
        position = transformed_action[:3]
        euler_angles = transformed_action[3:6]
        grip = transformed_action[6]

        self.trajectory['timesteps'].append(self.timestep)
        self.trajectory['positions'].append(position)
        self.trajectory['euler_angles'].append(euler_angles)
        self.trajectory['grips'].append(grip)

        self.timestep += 1

    def reset(self, init_state):
        self.__init__(init_state, self.transform_type)


class Visualizer:
    """
    A class to visualize images and trajectories.
    It creates a grid of images and plots the corresponding trajectories in 2D and 3D.

    Attributes:
        image_names: List of names for the images to be displayed.
        traj_names: List of names for the trajectories to be plotted.
        recoders: List of TrajectoryRecorder instances for each trajectory.
        base_width: Width of the base figure.
        base_height: Height of the base figure.
    
    Examples:
        ```python
        visualizer = Visualizer(
            image_names=['front', 'left_wrist', 'right_wrist'],
            traj_names=['left_arm', 'right_arm'],
            recoders=[TrajectoryRecorder(init_state=np.zeros(7), transform_type='ee_absolute') for _ in range(2)],
        )

        for sample in dataset:
            image1, image2, image3 = sample['observation.images.front'], sample['observation.images.left_wrist'], sample['observation.images.right_wrist']
            action1, action2 = sample['actions'][:7], sample['actions'][7:]
            visualizer.add(images=[image1, image2, image3], actions=[action1, action2])
            visualizer.plot()
        ```
    """

    def __init__(self, image_names, traj_names, recoders, base_width=5, base_height=5):
        self.image_names = image_names
        self.traj_names = traj_names
        self.recorders = recoders
        self.base_width = base_width
        self.base_height = base_height

        self.num_images = len(image_names)
        if self.num_images < 4:
            self.num_images = 4

        self.images = None
        self.fig = None

    def add(self, images, actions):
        self.images = images
        for recorder, action in zip(self.recorders, actions):
            recorder.add(action)
    
    def create_plot(self):
        plt.ion()
        self.fig = plt.figure(figsize=(self.base_width * self.num_images, self.base_height * 2))

    def plot(self):
        if self.fig is None:
            self.create_plot()
        
        plt.clf()
        for i, image in enumerate(self.images):
            ax = self.fig.add_subplot(2, self.num_images, i + 1)
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(self.image_names[i])

        # x
        ax1 = self.fig.add_subplot(2, self.num_images, self.num_images + 1)
        ax2 = self.fig.add_subplot(2, self.num_images, self.num_images + 2)
        ax3 = self.fig.add_subplot(2, self.num_images, self.num_images + 3)
        ax4 = self.fig.add_subplot(2, self.num_images, self.num_images + 4, projection='3d')

        for name, recoder in zip(self.traj_names, self.recorders):
            positions = np.array(recoder.trajectory['positions'])

            ax1.plot(positions[:, 0], positions[:, 1], label=name, linewidth=2)
            ax2.plot(positions[:, 1], positions[:, 2], label=name, linewidth=2)
            ax3.plot(positions[:, 0], positions[:, 2], label=name, linewidth=2)

            ax4.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                     label=name, linewidth=2)
            ax4.plot(positions[0, 0], positions[0, 1], positions[0, 2], 
                     'go', markersize=8)
            ax4.plot(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                     'ro', markersize=8)
            
            n_points = len(positions)
            step = max(1, n_points // 20) 
            euler_angles = np.array(recoder.trajectory['euler_angles'])

            for i in range(0, n_points, step):
                if i < n_points:
                    pos = positions[i]
                    direction = Rotation.from_euler('xyz', euler_angles[i]).apply([1, 0, 0])
                    ax4.quiver(pos[0], pos[1], pos[2], 
                              direction[0], direction[1], direction[2],
                              length=0.1, color='r', alpha=0.7)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
        ax1.axis('equal')
        ax2.set_xlabel('Y')
        ax2.set_ylabel('Z')
        ax2.legend()
        ax2.axis('equal')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.legend()
        ax3.axis('equal')

        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z')
        ax4.legend()
        plt.tight_layout()
        plt.axis('equal')
        plt.pause(1e-4)
    
    def reset(self, init_states):
        for recorder, init_state in zip(self.recorders, init_states):
            recorder.reset(init_state)
        self.images = None
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


def get_visualizer(image_names: list, traj_names: list, init_states: list, transform_type: str='ee_absolute') -> Visualizer:
    """
    Factory function to create a Visualizer instance with specified image names, trajectory names, and initial states.

    Args:
        image_names (list): List of names for the images to be displayed.
        traj_names (list): List of names for the trajectories to be plotted.
        init_states (list): List of initial states for the TrajectoryRecorder instances, each state should be a list of length (7,).
        transform_type (str): Type of transformation to apply to the actions. Defaults to 'ee_absolute'.
    """

    recorders = [TrajectoryRecorder(init_state, transform_type) for init_state in init_states]
    visualizer = Visualizer(image_names, traj_names, recorders)
    return visualizer