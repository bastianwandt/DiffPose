import torch


class Metrics:
    def __init__(self, init=0):
        self.init = init

    def mpjpe(self, p_ref, p, use_scaling=True, root_joint=0, num_joints=17):

        p = p.clone().reshape(-1, 3, num_joints)
        p_ref = p_ref.clone().reshape(-1, 3, num_joints)

        p = p - p[:, :, root_joint:root_joint+1]
        p_ref = p_ref - p_ref[:, :, root_joint:root_joint+1]

        if use_scaling:
            scale_p = p.reshape(-1, 3*num_joints).norm(p=2, dim=1, keepdim=True)
            scale_p_ref = p_ref.reshape(-1, 3*num_joints).norm(p=2, dim=1, keepdim=True)
            scale = scale_p_ref/scale_p
            p = (p.reshape(-1, 3*num_joints) * scale).reshape(-1, 3, num_joints)

        err = (p - p_ref).norm(p=2, dim=1).mean(axis=1)

        return err

    def PCK(self, p_ref, p, use_scaling=True, root_joint=6, num_joints=16, thresh=150.0):

        p = p.clone().reshape(-1, 3, num_joints)
        p_ref = p_ref.clone().reshape(-1, 3, num_joints)

        p = p - p[:, :, root_joint:root_joint+1]
        p_ref = p_ref - p_ref[:, :, root_joint:root_joint+1]

        if use_scaling:
            scale_p = p.reshape(-1, 3*num_joints).norm(p=2, dim=1, keepdim=True)
            scale_p_ref = p_ref.reshape(-1, 3*num_joints).norm(p=2, dim=1, keepdim=True)
            scale = scale_p_ref/scale_p
            p = (p.reshape(-1, 3*num_joints) * scale).reshape(-1, 3, num_joints)

        err = ((p - p_ref).norm(dim=1) < thresh).sum()/(p_ref.shape[0]*num_joints)*100

        return err

    def AUC(self, p_ref, p, use_scaling=True, root_joint=6, num_joints=16):

        p = p.clone().reshape(-1, 3, num_joints)
        p_ref = p_ref.clone().reshape(-1, 3, num_joints)

        p = p - p[:, :, root_joint:root_joint+1]
        p_ref = p_ref - p_ref[:, :, root_joint:root_joint+1]

        if use_scaling:
            scale_p = p.reshape(-1, 3*num_joints).norm(p=2, dim=1, keepdim=True)
            scale_p_ref = p_ref.reshape(-1, 3*num_joints).norm(p=2, dim=1, keepdim=True)
            scale = scale_p_ref/scale_p
            p = (p.reshape(-1, 3*num_joints) * scale).reshape(-1, 3, num_joints)

        distances = (p - p_ref).norm(dim=1)

        err = 0
        for t in torch.linspace(0, 150, 31):
            err += (distances < t).sum() / (distances.shape[0] * distances.shape[1] * 31)

        return err

    def pmpjpe(self, p_ref, p, use_reflection=True, use_scaling=True, num_joints=17):

        p = p.clone().reshape(-1, 3, num_joints)
        p_ref = p_ref.clone().reshape(-1, 3, num_joints)

        p_aligned = self.procrustes(p, p_ref, use_reflection=use_reflection, use_scaling=use_scaling)

        err = (p_ref - p_aligned).norm(p=2, dim=1).mean(axis=1)

        return err

    def procrustes(self, poses_inp, template_poses, use_reflection=True, use_scaling=True):

        num_joints = int(poses_inp.shape[-1])

        # translate template
        translation_template = template_poses.mean(axis=2, keepdims=True)
        template_poses_centered = template_poses - translation_template

        # scale template
        scale_t = torch.sqrt((template_poses_centered**2).sum(axis=[1, 2], keepdim=True) / (3*num_joints))
        template_poses_scaled = template_poses_centered / scale_t

        # translate prediction
        translation = poses_inp.mean(axis=2, keepdims=True)
        poses_centered = poses_inp - translation

        # scale prediction
        scale_p = torch.sqrt((poses_centered**2).sum(axis=[1, 2], keepdim=True) / (3*num_joints))
        poses_scaled = poses_centered / scale_p

        # rotation
        U, S, V = torch.svd(torch.matmul(template_poses_scaled, poses_scaled.transpose(2, 1)))
        R = torch.matmul(U, V.transpose(2, 1))

        # avoid reflection
        if not use_reflection:
            # only rotation
            Z = torch.eye(3).repeat(R.shape[0], 1, 1).to(poses_inp.device)
            Z[:, -1, -1] *= R.det()
            R = Z.matmul(R)

        poses_pa = torch.matmul(R, poses_scaled)

        # upscale again
        if use_scaling:
            poses_pa *= scale_t

        poses_pa += translation_template

        return poses_pa
