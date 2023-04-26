import numpy as np


def normalize_head(poses_2d, root_joint=0):
    # center at root joint
    p2d = poses_2d.reshape(-1, 2, 17)
    p2d -= p2d[:, :, [root_joint]]

    scale = np.linalg.norm(p2d[:, :, 0] - p2d[:, :, 10], axis=1, keepdims=True)
    p2ds = poses_2d / scale.mean()

    p2ds = p2ds * (1 / 10)

    return p2ds


def normalize_norm(poses_2d, root_joint=0):
    # center at root joint
    p2d = poses_2d.reshape(-1, 2, 17)
    p2d -= p2d[:, :, [root_joint]]

    p2d = p2d.reshape(-1, 34)

    scale = np.linalg.norm(p2d, axis=1, keepdims=True)
    p2ds = poses_2d / scale

    p2ds = p2ds

    return p2ds





