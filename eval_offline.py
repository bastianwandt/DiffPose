import numpy as np
import torch
import math
from einops import repeat, rearrange

from utils.eval_functions import (compute_CP_list, pa_hypo_batch,
                                  err_3dpe_parallel, compute_3DPCK,
                                  calculate_ece, calculate_symmetry_error)


def evaluate_metrics_over_hypotheses(gt, pred, batch_size=64):
    num_batches = math.ceil(gt.shape[0] / batch_size)
    n_hypo = pred.shape[1]
    num_samples = gt.shape[0]
    print(num_batches, num_samples)

    total_err_best_p1 = [[] for _ in range(n_hypo)]
    total_err_best_p2 = [[] for _ in range(n_hypo)]

    for i in range(num_batches):
        x_gt = gt[i*batch_size:(i+1)*batch_size]
        poses_3d_pred = pred[i*batch_size:(i+1)*batch_size]

        errors_proto1 = torch.mean(torch.sqrt(torch.sum((x_gt.reshape(x_gt.shape[0], 1, 3, 17)
                                                         - poses_3d_pred) ** 2, dim=2)), dim=2)
        x_gt = repeat(x_gt, 'b d j -> b n_hypo d j', n_hypo=n_hypo, d=3, j=17)
        errors_proto2 = err_3dpe_parallel(x_gt, poses_3d_pred.clone(), return_sum=False).reshape(x_gt.shape[0], -1)

        # best hypos
        for n in range(n_hypo):
            values, _ = torch.min(errors_proto1[:, :n+1], dim=1)
            total_err_best_p1[n].append(torch.sum(values).item())

            values, _ = torch.min(errors_proto2[:, :n+1], dim=1)
            total_err_best_p2[n].append(torch.sum(values).item())

    # total_err_best_p1 = [err / gt.shape[0] for err in total_err_best_p1]
    # total_err_best_p2 = [err / gt.shape[0] for err in total_err_best_p2]

    # print(total_err_best_p1)
    # print(total_err_best_p2)

    return total_err_best_p1, total_err_best_p2, num_samples


def evaluate_poses_in_file(fn, fn_gt, bs=128):
    poses = torch.from_numpy(np.load(fn))
    gt = torch.from_numpy(np.load(fn_gt))[0]

    if gt.dim() == 4:
        gt = gt.unsqueeze(0)

    if gt.shape[0] == poses.shape[1]:
        poses = poses.permute(1, 0, 2, 3)

    return evaluate_metrics_over_hypotheses(gt, poses, bs)


def evaluate_poses_in_single_file(fn, bs=128):
    poses = torch.from_numpy(np.load(fn, allow_pickle=True)['p3d_pred_sharma'].astype(np.float32, copy=False))
    gt = torch.from_numpy(np.load(fn, allow_pickle=True)['p3d_gt'].astype(np.float32, copy=False))
    if gt.dim() == 4:
        gt = gt.unsqueeze(0)

    if gt.shape[0] == poses.shape[1]:
        poses = poses.permute(1, 0, 2, 3)

    gt = gt.permute(0, 2, 1)
    poses = poses.permute(0, 1, 3, 2)

    return evaluate_metrics_over_hypotheses(gt, poses, bs)


def fuse_metrics_per_pose(metric_1, metric_2):
    max_hypo_res = metric_1[-1]

    for i in range(len(metric_2)):
        new_metric_2 = [min(m1, m2) for m1, m2 in zip(max_hypo_res, metric_2[i])]
        metric_1.append(new_metric_2)

    return metric_1


if __name__ == '__main__':
    bs = 64

    fn = '/home/karl/Projects/probabilistic_human_pose/codebase/diffusion/results/sharma_hard_set_results1000hypos.pickle'
    mpjpe_per_pose, pmpjpe_per_pose, num_samples = evaluate_poses_in_single_file(fn, bs)
    mpjpe_sharma = [np.sum(m) / num_samples for m in mpjpe_per_pose]
    pmpjpe_sharma = [np.sum(m) / num_samples for m in pmpjpe_per_pose]
    print(mpjpe_sharma)
    print(pmpjpe_sharma)

    fn_gt = '/home/karl/Projects/probabilistic_human_pose/codebase/diffusion/results/ground_truth_poses.npy'
    fn = '/home/karl/Projects/probabilistic_human_pose/codebase/diffusion/results/wehrbein_hard_set_results_1000hypos_part1.npy'
    mpjpe_per_pose_1, pmpjpe_per_pose_1, num_samples = evaluate_poses_in_file(fn, fn_gt, bs)

    fn = '/home/karl/Projects/probabilistic_human_pose/codebase/diffusion/results/wehrbein_hard_set_results_1000hypos_part1.npy'
    mpjpe_per_pose_2, pmpjpe_per_pose_2, num_samples = evaluate_poses_in_file(fn, fn_gt, bs)

    mpjpe_per_pose = fuse_metrics_per_pose(mpjpe_per_pose_1, mpjpe_per_pose_2)
    pmpjpe_per_pose = fuse_metrics_per_pose(pmpjpe_per_pose_1, pmpjpe_per_pose_2)

    mpjpe_wehrbein = [np.sum(m) / num_samples for m in mpjpe_per_pose]
    pmpjpe_wehrbein = [np.sum(m) / num_samples for m in pmpjpe_per_pose]

    print(mpjpe_wehrbein)
    print(pmpjpe_wehrbein)

    print(len(mpjpe_wehrbein))






