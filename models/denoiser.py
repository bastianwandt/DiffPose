import torch.nn.functional as F
import torch.nn as nn
import torch
from einops import rearrange, repeat, reduce


class Concat(nn.Module):
    """
    Basic denoiser network which concatenates all noisy 3d pose, time step, and conditioning vector and passes it through
    a small residual network to generate a denoised 3d pose
    """
    def __init__(self, num_joints: int = 16, cond_dim: int = 32, num_samples: int = 16, time_embedding: str = None):
        super().__init__()

        self.num_joints = num_joints
        self.num_samples = num_samples

        if cond_dim < 0:
            self.cond_dim = num_joints*num_samples*2
        else:
            self.cond_dim = cond_dim

        if time_embedding is None:
            self.t_dim = 1
        else:
            raise NotImplementedError("time embedding not implemented yet")

        self.upscale = nn.Linear(num_joints*3 + self.cond_dim + self.t_dim, 1024)
        self.res_pose1 = ResBlock()
        self.res_pose2 = ResBlock()
        self.pose = nn.Linear(1024, num_joints*3)

    def forward(self, x, t, c):
        b, _ = c.shape
        t = repeat(t, 'b -> b 1')

        x = self.upscale(torch.cat((x, t, c), dim=-1))
        xp = self.res_pose1(x)
        xp = self.res_pose2(xp)
        xp = self.pose(xp)

        return xp

#
# class pose_denoiser(nn.Module):
#     def __init__(self, num_joints=16, cond_dim=32, time_embedding=None):
#         super(pose_denoiser, self).__init__()
#         self.cond_dim = cond_dim
#         self.num_joints = num_joints
#
#         if time_embedding is None:
#             self.t_dim = 1
#         else:
#             raise NotImplementedError("time embedding not implemented yet")
#
#         # self.upscale = nn.Linear(51+1+34, 1024)
#         self.upscale = nn.Linear(num_joints*3+self.t_dim+cond_dim, 1024)
#         self.res_pose1 = ResBlock()
#         self.res_pose2 = ResBlock()
#         self.pose = nn.Linear(1024, num_joints*3)
#
#     def forward(self, x, t, c):
#         b, _ = c.shape
#         t = repeat(t, 'b -> b 1')
#
#         x = self.upscale(torch.cat((x, t, c), dim=-1))
#         xp = self.res_pose1(x)
#         xp = self.res_pose2(xp)
#         xp = self.pose(xp)
#
#         return xp


class ResBlock(nn.Module):
    """
    Residual block
    """
    def __init__(self, num_neurons: int = 1024, use_batchnorm: bool = False):
        super(ResBlock, self).__init__()

        self.use_batchnorm = use_batchnorm
        self.l1 = nn.Linear(num_neurons, num_neurons)
        self.bn1 = nn.BatchNorm1d(num_neurons)
        self.l2 = nn.Linear(num_neurons, num_neurons)
        self.bn2 = nn.BatchNorm1d(num_neurons)

    def forward(self, x):
        inp = x
        x = F.leaky_relu(self.l1(x))
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.leaky_relu(self.l2(x))
        if self.use_batchnorm:
            x = self.bn2(x)
        x += inp

        return x


def get_denoiser(cfg):
    """
    Factory function for creating the denoiser based on the config file
    """
    if cfg.MODEL.DENOISER_TYPE == 'base':
        if cfg.MODEL.CONDITION_TYPE == 'embedded_poseformer':
            # Embedded_poseformer flattens the individual joint embeddings into one vector
            cond_dim = cfg.DATASET.COND_JOINTS * cfg.MODEL.CONDITION_DIM
        else:
            cond_dim = cfg.MODEL.CONDITION_DIM
        njoints = cfg.DATASET.NUM_JOINTS
        """Handles non H3.6m datasets by also predicting the root joint"""
        if njoints != 16:
            njoints += 1

        model = Concat(
            num_joints=njoints,
            cond_dim=cond_dim
        )
    elif cfg.MODEL.DENOISER_TYPE == 'base_concat':
          model = Concat(
            num_joints=cfg.DATASET.NUM_JOINTS,
            cond_dim=cfg.MODEL.CONDITION_DIM,
            num_samples=cfg.MODEL.EXTRA.NUM_SAMPLES
    )
    else:
        raise RuntimeError("Incorrect denoiser type, base implemented, tried to use: ", cfg.MODEL.DENOISER_TYPE)

    return model



