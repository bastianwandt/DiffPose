import torch
from torch.utils import data

from models.trainer import Trainer
from models.diffusion import GaussianDiffusion
from models.denoiser import get_denoiser
from models.condition_embedding import get_condition_embedding

import config as c
from data.data_h36m import H36MDataset, H36MDatasetH5
from utils.data_utils import H36M_NAMES, MPII_NAMES, H36M17j_TO_MPII, get_symmetry

from experiments.default import get_config_file


def project_hard_samples(trainer):
    if trainer.config.DATASET.DATA_FORMAT == 'pickle':
        test_dataset = H36MDataset(
            trainer.config.DATASET.ROOT + 'testset_h36m.pickle', train_set=False,
            quick_eval=False,
            hardsubset=True
        )
    elif trainer.config.DATASET.DATA_FORMAT == 'h5':
        test_dataset = H36MDatasetH5(
            trainer.config.DATASET.ROOT + 'dataset_test.h5', train_set=False,
            quick_eval=False,
            hardsubset=True
        )
    else:
        raise RuntimeError(
            "Incorrect data format choice for dataset, available [h5, pickle], attempted: {}".format(
                trainer.config.DATASET.DATA_FORMAT))

    loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    print(len(loader))

    for sample in loader:
        heatmap = sample["heatmaps"].cpu()
        pred_z0, pred_samples, gt = trainer.predict(sample, n_hypo=200)

        symmetry_gt = get_symmetry(gt, reduction='mean')
        symmetry_z0 = get_symmetry(pred_z0, reduction='mean')
        symmetry_samples = get_symmetry(pred_samples, reduction='mean')

        print('Delta bone length (mm) GT:', symmetry_gt)
        print('Delta bone length (mm) z0:', symmetry_z0)
        print('Delta bone length (mm) samples:', symmetry_samples)

        print(pred_z0.shape, pred_samples.shape)

        sampled_poses_numpy = pred_samples.permute(1, 0, 2, 3).cpu().numpy()
        gt_poses_numpy = gt.unsqueeze(0).cpu().numpy()

        import numpy as np
        np.save('ours_sampled_poses', sampled_poses_numpy)
        np.save('ground_truth_poses', gt_poses_numpy)

        import matplotlib.pyplot as plt

        x = pred_samples[0, :, 0]
        y = -pred_samples[0, :, 1]

        x_gt = gt[0, 0, :]
        y_gt = -gt[0, 1, :]

        x_max = x_gt.max()
        x_min = x_gt.min()

        x_center = (x_max + x_min) / 2
        x_diff = x_max - x_min

        y_min = y_gt.max()
        y_max = y_gt.min()
        y_center = (y_max + y_min) / 2
        y_diff = y_max - y_min

        scale = max(x_diff.abs(), y_diff.abs()) * 1.2

        x_coords = (x - x_center) / scale
        y_coords = (y - y_center) / scale

        print(x_center, y_center)
        print(x_coords.max() - x_coords.min())
        print(y_coords.max() - y_coords.min())

        plt.imshow(heatmap[0].sum(dim=0), vmin=0., vmax=1.)
        plt.colorbar()
        plt.scatter(
            32 + 64 * x_coords,
            32 + 64 * y_coords,
            1.,
            'r'
        )
        plt.show()

        for i in range(16):
            hist = torch.histogramdd(
                32 + 64 * torch.stack(
                    (y_coords[:, H36M17j_TO_MPII[i]], x_coords[:, H36M17j_TO_MPII[i]]),
                    dim=1),
                bins=(64, 64),
                range=(0, 65, 0, 65)
            )
            plt.imshow(hist[0])
            plt.show()
            exit()

            plt.imshow(heatmap[0, i], vmin=0., vmax=1.)
            plt.colorbar()

            plt.scatter(
                32 + 64*x_coords[:, H36M17j_TO_MPII[i]],
                32 + 64*y_coords[:, H36M17j_TO_MPII[i]],
                1.,
                'r'
            )
            plt.show()
        exit()



if __name__ == '__main__':
    import argparse

    baseline_config = 'results_add/config.yaml'
    # baseline_config = 'experiments/h36m/heatmap_sampled.yaml'
    # baseline_config = 'experiments/h36m/priorgrad_mean_only.yaml'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=baseline_config,
                        help='Path to experiment config file to use')
    parser.add_argument('--do_not_use_wandb', action='store_true', default=False,
                        help='Deactivates the use of wandb')
    args = parser.parse_args()

    cfg = get_config_file(args.config)

    wandb_config = {
        'Dataset': cfg.DATASET.DATASET,
        'Dataformat': cfg.DATASET.DATA_FORMAT,

        'Diffusion type': cfg.MODEL.DIFFUSION_TYPE,
        'Diffusion prior': cfg.MODEL.DIFFUSION_PRIOR,
        'Diffusion prior key': cfg.MODEL.DIFFUSION_PRIOR_KEY,

        'Condition type': cfg.MODEL.CONDITION_TYPE,
        'Condition key': cfg.MODEL.CONDITION_KEY,

        'Denoiser type': cfg.MODEL.DENOISER_TYPE,

        'Loss type': cfg.LOSS.LOSS_TYPE,
        'Diffusion objective': cfg.MODEL.DIFFUSION_OBJECTIVE,
        'Timesteps': cfg.MODEL.TIMESTEPS,
        'Iterations': cfg.TRAIN.NUM_STEPS,

        'LR': cfg.TRAIN.LR
    }
    import wandb

    denoiser = get_denoiser(cfg)
    conditioner = get_condition_embedding(cfg)

    diffusion = GaussianDiffusion(
        denoiser,
        conditioner,
        image_size=128,
        objective=cfg.MODEL.DIFFUSION_OBJECTIVE,
        timesteps=cfg.MODEL.TIMESTEPS,   # number of steps
        loss_type=cfg.LOSS.LOSS_TYPE,    # L1 or L2
        scaling_3d_pose=cfg.TRAIN.SCALE_3D_POSE,  # Pre-conditioning of target scale
        noise_scale=cfg.MODEL.NOISE_SCALE,
        cosine_offset=cfg.TRAIN.COSINE_OFFSET
    ).cuda()

    from pathlib import Path
    results_folder = Path(args.config).parent.__str__()

    trainer = Trainer(
        diffusion,
        None,
        train_batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        train_lr=cfg.TRAIN.LR,
        train_num_steps=cfg.TRAIN.NUM_STEPS,         # total training steps
        gradient_accumulate_every=cfg.TRAIN.GRADIENT_ACCUMULATE_EVERY,    # gradient accumulation steps
        ema_decay=cfg.TRAIN.EMA_DECAY,  # exponential moving average decay
        amp=cfg.TRAIN.AMP,  # turn on mixed precision
        condition_type=cfg.MODEL.CONDITION_TYPE,
        prior_type=cfg.MODEL.DIFFUSION_PRIOR,
        results_folder=results_folder,
        config=cfg,
        use_wandb=True
    )

    trainer.load('final')

    project_hard_samples(trainer)

    print('done')

