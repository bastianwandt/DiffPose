import torch
from torch.utils import data

from models.trainer import Trainer
from models.diffusion import GaussianDiffusion
from models.denoiser import get_denoiser
from models.condition_embedding import get_condition_embedding
import numpy as np

import config as c
from data.data_h36m import H36MDatasetH5, HPDatasetH5

from experiments.default import get_config_file


def generate_posefile_3dhp(trainer):
    actions = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']

    for action_idx, action in enumerate(actions):
        dataset_name = trainer.config.DATASET.TESTFILE
        test_dataset = HPDatasetH5(
            trainer.config.DATASET.ROOT + dataset_name, train_set=False,
            quick_eval=False,
            actions=[action], hardsubset=False, use_validation=False
        )
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

        samples_z0 = []
        samples_poses = []
        samples_gt = []
        for sample in loader:
            z0, poses, gt = trainer.predict(sample, n_hypo=200)
            samples_z0.append(z0.cpu().numpy())
            samples_poses.append(poses.cpu().numpy())
            samples_gt.append(gt.cpu().numpy())

        np.save("{}/{}_samples_z0.npy".format(trainer.results_folder, action), samples_z0, allow_pickle=True)
        np.save("{}/{}_samples_poses.npy".format(trainer.results_folder, action), samples_poses, allow_pickle=True)
        np.save("{}/{}_samples_gt.npy".format(trainer.results_folder, action), samples_gt, allow_pickle=True)


def generate_posefile_h36m_full_trainset(trainer, n_hypo=200, action_wise=False):
    actions = [
        'Directions', 'Discussion', 'Eating', 'Greeting',
        'Phoning', 'Photo', 'Posing', 'Purchases', 'Sitting',
        'SittingDown', 'Smoking', 'Waiting', 'WalkDog',
        'WalkTogether', 'Walking'
    ]
    if action_wise:
        for action_idx, action in enumerate(actions):
            dataset_name = trainer.config.DATASET.TRAINFILE
            train_dataset = H36MDatasetH5(
                trainer.config.DATASET.ROOT + dataset_name, train_set=True,
                quick_eval=True,
                actions=[action], hardsubset=False, use_validation=False
            )
            loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=False)

            print('Samples in dataset {}: {}'.format(action, len(train_dataset)), flush=True)
            print('Samples in loader', len(loader))
            samples_z0 = []
            samples_poses = []
            samples_gt = []
            for sample in loader:
                z0, poses, gt = trainer.predict(sample, n_hypo=n_hypo)
                for s in zip(z0, poses, gt):
                    samples_z0.append(s[0].cpu().numpy())
                    samples_poses.append(s[1].cpu().numpy())
                    samples_gt.append(s[2].cpu().numpy())

            np.save("{}/{}_train_samples_z0.npy".format(trainer.results_folder, action), samples_z0, allow_pickle=True)
            np.save("{}/{}_train_samples_poses.npy".format(trainer.results_folder, action), samples_poses, allow_pickle=True)
            np.save("{}/{}_train_samples_gt.npy".format(trainer.results_folder, action), samples_gt, allow_pickle=True)
    else:
        dataset_name = trainer.config.DATASET.TRAINFILE
        train_dataset = H36MDatasetH5(
            trainer.config.DATASET.ROOT + dataset_name, train_set=True,
            quick_eval=True, quick_eval_stride=16,
            actions=actions, hardsubset=False, use_validation=False
        )
        loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=False)

        print('Samples in dataset', len(train_dataset), flush=True)
        print('Samples in loader', len(loader), flush=True)
        samples_z0 = []
        samples_poses = []
        samples_gt = []
        for i, sample in enumerate(loader):
            print('Processing: {}/{}'.format(i, len(loader)), flush=True)
            z0, poses, gt = trainer.predict(sample, n_hypo=n_hypo)
            for s in zip(z0, poses, gt):
                samples_z0.append(s[0].cpu().numpy())
                samples_poses.append(s[1].cpu().numpy())
                samples_gt.append(s[2].cpu().numpy())

        np.save("{}/{}_train_samples_z0.npy".format(trainer.results_folder, 'full'), samples_z0, allow_pickle=True)
        np.save("{}/{}_train_samples_poses.npy".format(trainer.results_folder, 'full'), samples_poses,
                allow_pickle=True)
        np.save("{}/{}_train_samples_gt.npy".format(trainer.results_folder, 'full'), samples_gt, allow_pickle=True)


def generate_posefile_h36m_full(trainer, n_hypo=200, action_wise=False):
    actions = [
        'Directions', 'Discussion', 'Eating', 'Greeting',
        'Phoning', 'Photo', 'Posing', 'Purchases', 'Sitting',
        'SittingDown', 'Smoking', 'Waiting', 'WalkDog',
        'WalkTogether', 'Walking'
    ]
    if action_wise:
        for action_idx, action in enumerate(actions):
            dataset_name = trainer.config.DATASET.TESTFILE
            test_dataset = H36MDatasetH5(
                trainer.config.DATASET.ROOT + dataset_name, train_set=False,
                quick_eval=False,
                actions=[action], hardsubset=False, use_validation=False
            )
            loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

            print('Samples in dataset {}', len(test_dataset), action)
            print('Samples in loader', len(loader))
            samples_z0 = []
            samples_poses = []
            samples_gt = []
            for sample in loader:
                z0, poses, gt = trainer.predict(sample, n_hypo=n_hypo)
                samples_z0.append(z0.cpu().numpy())
                samples_poses.append(poses.cpu().numpy())
                samples_gt.append(gt.cpu().numpy())

            np.save("{}/{}_samples_z0.npy".format(trainer.results_folder, action), samples_z0, allow_pickle=True)
            np.save("{}/{}_samples_poses.npy".format(trainer.results_folder, action), samples_poses, allow_pickle=True)
            np.save("{}/{}_samples_gt.npy".format(trainer.results_folder, action), samples_gt, allow_pickle=True)
    else:
        dataset_name = trainer.config.DATASET.TESTFILE
        test_dataset = H36MDatasetH5(
            trainer.config.DATASET.ROOT + dataset_name, train_set=False,
            quick_eval=False,
            actions=actions, hardsubset=False, use_validation=False
        )
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

        print('Samples in dataset', len(test_dataset), flush=True)
        print('Samples in loader', len(loader), flush=True)
        samples_z0 = []
        samples_poses = []
        samples_gt = []
        for i, sample in enumerate(loader):
            print('Processing: {}/{}'.format(i, len(loader)), flush=True)
            z0, poses, gt = trainer.predict(sample, n_hypo=n_hypo)
            for s in zip(z0, poses, gt):
                samples_z0.append(s[0].cpu().numpy())
                samples_poses.append(s[1].cpu().numpy())
                samples_gt.append(s[2].cpu().numpy())

        np.save("{}/{}_samples_z0.npy".format(trainer.results_folder, 'full'), samples_z0, allow_pickle=True)
        np.save("{}/{}_samples_poses.npy".format(trainer.results_folder, 'full'), samples_poses, allow_pickle=True)
        np.save("{}/{}_samples_gt.npy".format(trainer.results_folder, 'full'), samples_gt, allow_pickle=True)


def generate_posefile_h36m_hard(trainer, n_hypo=200):

    dataset_name = trainer.config.DATASET.TESTFILE
    test_dataset = H36MDatasetH5(
        trainer.config.DATASET.ROOT + dataset_name, train_set=False,
        quick_eval=False, hardsubset=True, use_validation=False
    )
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    print('Samples in dataset', len(test_dataset), flush=True)
    print('Samples in loader', len(loader), flush=True)
    samples_z0 = []
    samples_poses = []
    samples_gt = []
    for sample in loader:
        z0, poses, gt = trainer.predict(sample, n_hypo=n_hypo)
        samples_z0.append(z0.cpu().numpy())
        samples_poses.append(poses.cpu().numpy())
        samples_gt.append(gt.cpu().numpy())

    print('Num samples to save ', len(samples_poses))

    np.save("{}/{}_samples_z0.npy".format(trainer.results_folder, "hard"), samples_z0, allow_pickle=True)
    np.save("{}/{}_samples_poses.npy".format(trainer.results_folder, "hard"), samples_poses, allow_pickle=True)
    np.save("{}/{}_samples_gt.npy".format(trainer.results_folder, "hard"), samples_gt, allow_pickle=True)


def generate_posefile(trainer, n_hypo=200):
    if trainer.config.DATASET.DATASET == '3DHP':
        generate_posefile_3dhp(trainer)
    else:
        print('Generating poses for trainset of H36M', flush=True)
        generate_posefile_h36m_full_trainset(trainer, n_hypo=n_hypo)
        print('Generating poses for testset of H36M', flush=True)
        generate_posefile_h36m_full(trainer, n_hypo=n_hypo)
        print('Generating poses for hard subset of H36M', flush=True)
        generate_posefile_h36m_hard(trainer, n_hypo=n_hypo)


if __name__ == '__main__':
    import argparse

    baseline_config = 'experiments/h36m/baseline.yaml'
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

        'LR': cfg.TRAIN.LR,

        'config_path': args.config
    }

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

    # train_dataset = H36MDataset(c.data_base_dir + 'trainset_h36m.pickle', train_set=True)
    # test_dataset = H36MDataset(c.data_base_dir + 'testset_h36m.pickle', train_set=False)
    #
    # train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    # test_loader = data.DataLoader(test_dataset, batch_size=10)
    # results_folder = "{}_{}".format(cfg.OUTPUT_DIR, run_name)
    # results_folder = cfg.OUTPUT_DIR
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

    print('Generating 20 hypotheses')
    generate_posefile(trainer, n_hypo=20)
