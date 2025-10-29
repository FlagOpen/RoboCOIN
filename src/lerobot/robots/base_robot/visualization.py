"""
Visualizer for robot trajectories and images.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from typing import List


class TrajectoryRecorder:
    """
    Records the trajectory of a robot's end-effector.
    Stores positions, euler angles, and grip states over time.
    e.g.
    ```python
    recorder = TrajectoryRecorder()
    recorder.add(action)  # action is a list or array of [x, y, z, roll, pitch, yaw, grip]
    trajectory = recorder.trajectory  # get the recorded trajectory
    ```
    """
    def __init__(self) -> None:
        self.trajectory = {
            'timesteps': [0],
            'positions': [],
            'euler_angles': [],
            'grips': [],
        }
        self.timestep = 0

    def add(self, action: List[float]) -> None:
        """
        Add a new action to the trajectory.
        Params:
        - action: list or array of [x, y, z, roll, pitch, yaw, grip]
        """
        assert len(action) == 7, "Action must have 7 elements: [x, y, z, roll, pitch, yaw, grip]"

        position = action[:3]
        euler_angles = action[3:6]
        grip = action[6]

        self.trajectory['timesteps'].append(self.timestep)
        self.trajectory['positions'].append(position)
        self.trajectory['euler_angles'].append(euler_angles)
        self.trajectory['grips'].append(grip)

        self.timestep += 1

    def reset(self) -> None:
        """
        Reset the trajectory recorder to initial state.
        """
        self.trajectory = {
            'timesteps': [0],
            'positions': [],
            'euler_angles': [],
            'grips': [],
        }
        self.timestep = 0


class Visualizer:
    """
    Visualizer for robot images and trajectories.
    The first row displays images from different cameras.
    The second row displays 2D projections of the trajectories (if draw_2d is True)
    and/or 3D plot of the trajectories (if draw_3d is True).
    Params:
    - image_names: list of str, names for the images to be displayed.
    - traj_names: list of str, names for the trajectories to be plotted.
    - recorders: list of TrajectoryRecorder, one for each trajectory.
    - base_width: int, base width for the figure.
    - base_height: int, base height for the figure.
    - draw_2d: bool, whether to draw 2D projections of the trajectories
    - draw_3d: bool, whether to draw 3D plots of the trajectories
    e.g.
    ```python
    visualizer = Visualizer(
        image_names=['Camera 1', 'Camera 2'],
        traj_names=['left', 'right'],
        recorders=[recorder1, recorder2],
        draw_2d=True,
        draw_3d=True,
    )
    visualizer.add(images, actions)  # images is a list of images, actions is a list of actions for each recorder
    visualizer.plot()  # plot the images and trajectories
    ```
    """

    def __init__(
            self, 
            image_names: List[str], 
            traj_names: List[str],
            recoders: List[TrajectoryRecorder],
            base_width: int = 5,
            base_height: int = 5,
            draw_2d: bool = True,
            draw_3d: bool = True,
        ) -> None:
        self.image_names = image_names
        self.traj_names = traj_names
        self.recorders = recoders
        self.base_width = base_width
        self.base_height = base_height
        self.draw_2d = draw_2d
        self.draw_3d = draw_3d

        # minimum 4 columns to keep layout consistent
        self.num_images = len(image_names)
        if self.num_images < 4:
            self.num_images = 4

        self.images = None
        self.fig = None

    def add(self, images: List[np.ndarray], actions: List[List[float]]) -> None:
        """
        Add new images and actions to the visualizer.
        Params:
        - images: list of images to be displayed.
        - actions: list of actions for each trajectory recorder.
        """
        self.images = images
        for recorder, action in zip(self.recorders, actions):
            recorder.add(action)
    
    def create_plot(self) -> None:
        """
        Create the matplotlib figure for plotting.
        row1: images
        row2: x-y, y-z, x-z projections (if draw_2d), 3D plot (if draw_3d)
        """
        plt.ion()
        self.fig = plt.figure(figsize=(self.base_width * self.num_images, self.base_height * 2))

    def plot(self) -> None:
        """
        Plot the images and trajectories.
        1st row: images from different cameras
        2nd row: 2D projections and/or 3D plot of trajectories
        """
        if self.fig is None:
            self.create_plot()
        
        plt.clf()

        # determine number of rows and columns
        nrows = 1
        if self.draw_2d:
            nrows += 1
        if self.draw_3d:
            nrows += 1
        ncols = max(self.num_images, 3)

        row = 0

        # plot images: [cam1, cam2, ..., camN]
        for i, image in enumerate(self.images):
            ax = self.fig.add_subplot(nrows, ncols, i + 1)
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(self.image_names[i])
        
        # plot 2d trajectories: x-y, y-z, x-z
        if self.draw_2d:
            row += 1
            ax1 = self.fig.add_subplot(nrows, ncols, ncols * row + 1)
            ax2 = self.fig.add_subplot(nrows, ncols, ncols * row + 2)
            ax3 = self.fig.add_subplot(nrows, ncols, ncols * row + 3)
        
        # plot 3d trajectories: x-y-z
        if self.draw_3d:
            row += 1
            ax4 = self.fig.add_subplot(nrows, ncols, ncols * row + 1, projection='3d')

        for name, recorder in zip(self.traj_names, self.recorders):
            positions = np.array(recorder.trajectory['positions'])

            # 2D plots
            if self.draw_2d:
                ax1.plot(positions[:, 0], positions[:, 1], label=name, linewidth=2)
                ax2.plot(positions[:, 1], positions[:, 2], label=name, linewidth=2)
                ax3.plot(positions[:, 0], positions[:, 2], label=name, linewidth=2)

            # 3D plot
            if self.draw_3d:
                # plot 3D trajectory
                ax4.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                        label=name, linewidth=2)
                # mark start and end points
                ax4.plot(positions[0, 0], positions[0, 1], positions[0, 2], 
                        'go', markersize=8)
                ax4.plot(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                        'ro', markersize=8)
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
            
                # plot orientation arrows every N points
                n_points = len(positions)
                step = max(1, n_points // 20) 
                euler_angles = np.array(recorder.trajectory['euler_angles'])

                for i in range(0, n_points, step):
                    if i < n_points:
                        # plot orientation arrows
                        pos = positions[i]
                        direction = Rotation.from_euler('xyz', euler_angles[i]).apply([1, 0, 0])
                        ax4.quiver(pos[0], pos[1], pos[2], 
                                direction[0], direction[1], direction[2],
                                length=0.1, color='r', alpha=0.7)

                ax4.set_xlabel('X')
                ax4.set_ylabel('Y')
                ax4.set_zlabel('Z')
                ax4.legend()

        plt.tight_layout()
        plt.axis('equal')
        plt.pause(1e-4) # short pause to update the plot
    
    def reset(self) -> None:
        """
        Reset the visualizer and all recorders.
        """
        for recorder in self.recorders:
            recorder.reset()
        self.images = None
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


def get_visualizer(
        image_names: List[str], 
        traj_names: List[str], 
        draw_2d: bool, 
        draw_3d: bool
    ) -> Visualizer:
    """
    Factory function to create a Visualizer instance with specified image names, 
    trajectory names, and visualization options.
    Params:
    - image_names (list): Names of camera views to display
    - traj_names (list): Names of trajectories to visualize
    - draw_2d (bool): Enable 2D trajectory projections
    - draw_3d (bool): Enable 3D trajectory visualization
    Returns:
    - Visualizer: Configured visualization instance
    """
    recorders = [TrajectoryRecorder() for _ in range(len(traj_names))]
    visualizer = Visualizer(image_names, traj_names, recorders, draw_2d=draw_2d, draw_3d=draw_3d)
    return visualizer