import torch


def get_bones_all(poses):
    bone_map = [[6, 0], [0, 1], [1, 2], [6, 3], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [7, 10], [10, 11], [11, 12],
                [7, 13], [13, 14], [14, 15]]

    poses = poses.reshape((-1, 3, 16))
    ext_bones = poses[:, :, bone_map]
    bones = ext_bones[:, :, :, 0] - ext_bones[:, :, :, 1]

    return bones


def get_bone_lengths(bones):
    bone_lengths = torch.norm(bones, p=2, dim=1)

    return bone_lengths


def get_bone_lengths_all(poses):
    bone_map = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12],
                [12, 13], [8, 14], [14, 15], [15, 16]]

    poses = poses.reshape((-1, 3, 17))

    ext_bones = poses[:, :, bone_map]

    bones = ext_bones[:, :, :, 0] - ext_bones[:, :, :, 1]

    bone_lengths = torch.norm(bones, p=2, dim=1)

    return bone_lengths

