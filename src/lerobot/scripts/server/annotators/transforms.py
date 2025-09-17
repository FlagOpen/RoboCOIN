import numpy as np
from scipy.spatial.transform import Rotation as R


def quaternion_to_euler(quat, order='xyz', scalar_first=False):
    r = R.from_quat(quat, scalar_first=scalar_first)
    return r.as_euler(order, degrees=True).tolist()


def euler_to_quaternion(euler, order='xyz', scalar_first=False):
    r = R.from_euler(order, euler, degrees=True)
    return r.as_quat(scalar_first=scalar_first).tolist()


def matrix_to_euler(rot_matrix, order='xyz'):
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    r = R.from_matrix(rot_matrix)
    return r.as_euler(order, degrees=True).tolist()

def euler_to_matrix(euler, order='xyz'):
    r = R.from_euler(order, euler, degrees=True)
    return r.as_matrix().tolist()


def matrix6d_to_euler(matrix6d, order='xyz'):
    matrix6d = np.array(matrix6d).reshape(3, 2)
    z_axis = np.cross(matrix6d[:, 0], matrix6d[:, 1])
    rot_matrix = np.stack([matrix6d[:, 0], matrix6d[:, 1], z_axis], axis=-1)
    r = R.from_matrix(rot_matrix)
    return r.as_euler(order, degrees=True).tolist()


def euler_to_matrix6d(euler, order='xyz'):
    r = R.from_euler(order, euler, degrees=True)
    rot_matrix = r.as_matrix()
    return rot_matrix[:, :2].flatten().tolist()


def euler_add(base_euler, add_euler, order='xyz'):
    r_base = R.from_euler(order, base_euler, degrees=True)
    r_add = R.from_euler(order, add_euler, degrees=True)
    r_combined = r_add * r_base
    return r_combined.as_euler(order, degrees=True).tolist()


def euler_subtract(base_euler, sub_euler, order='xyz'):
    r_base = R.from_euler(order, base_euler, degrees=True)
    r_sub = R.from_euler(order, sub_euler, degrees=True)
    r_combined = r_sub.inv() * r_base
    return r_combined.as_euler(order, degrees=True).tolist()


def position_add(base_pos, add_pos):
    return (np.array(base_pos) + np.array(add_pos)).tolist()


def position_subtract(base_pos, sub_pos):
    return (np.array(base_pos) - np.array(sub_pos)).tolist()


def position_rotate(base_pos, euler, order='xyz'):
    r = R.from_euler(order, euler, degrees=True)
    rotated_pos = r.apply(base_pos)
    return rotated_pos.tolist()