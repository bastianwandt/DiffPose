import torch
import wandb
import random
import numpy as np

from models.trainer import Trainer
from models.diffusion import GaussianDiffusion
from models.denoiser import get_denoiser
from models.condition_embedding import get_condition_embedding
from data.data_h36m import H36MDataset, H36MDatasetH5, PW3DDatasetH5
from experiments.default import get_config_file


if __name__ == '__main__':
    import argparse

    baseline_config = 'experiments/h36m/baseline.yaml'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=baseline_config,
                        help='Path to experiment config file to use')
    parser.add_argument('--do_not_use_wandb', action='store_true', default=False,
                        help='Deactivates the use of wandb')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for training')
    args = parser.parse_args()

    """Initialize random seeds"""
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = get_config_file(args.config)

    """Log some of the config settings to wandb if available"""
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
        'seed': args.seed
    }

    if not args.do_not_use_wandb:
        wandb.init(
            project="Train",
            name=cfg.MODEL.NAME,
            entity="probabilistic_human_pose",
            config=wandb_config
        )
        run_name = wandb.run.id
    else:
        """Use current time as run name if wandb is unavailable"""
        import time
        run_name = str(time.time_ns())

    results_folder = "{}_{}".format(cfg.OUTPUT_DIR, run_name)

    denoiser = get_denoiser(cfg)
    conditioner = get_condition_embedding(cfg)

    """Add root joint to joints to predict if not H3.6M skeleton"""
    num_joints = cfg.DATASET.NUM_JOINTS
    if num_joints != 16:
        num_joints += 1

    diffusion = GaussianDiffusion(
        denoiser,
        conditioner,
        objective=cfg.MODEL.DIFFUSION_OBJECTIVE,
        timesteps=cfg.MODEL.TIMESTEPS,   # number of steps
        loss_type=cfg.LOSS.LOSS_TYPE,    # L1 or L2
        scaling_3d_pose=cfg.TRAIN.SCALE_3D_POSE,  # Pre-conditioning of target scale
        noise_scale=cfg.MODEL.NOISE_SCALE,
        cosine_offset=cfg.TRAIN.COSINE_OFFSET,
        num_joints=num_joints
    ).cuda()

    num_parameters = sum(p.numel() for p in diffusion.parameters())
    num_parameters_denoiser = sum(p.numel() for p in denoiser.parameters())
    num_parameters_conditioner = sum(p.numel() for p in conditioner.parameters())
    print("Number of parameters of the model", num_parameters)
    print("Number of parameters of the model (denoiser)", num_parameters_denoiser)
    print("Number of parameters of the model (conditioner)", num_parameters_conditioner)
    if not args.do_not_use_wandb:
        wandb.run.summary["Num Parameters"] = num_parameters
        wandb.run.summary["Num Parameters (denoiser)"] = num_parameters_denoiser
        wandb.run.summary["Num Parameters (conditioner)"] = num_parameters_conditioner

    # Store config file in results folder for replication and eval
    import shutil
    from pathlib import Path
    Path(results_folder).mkdir(exist_ok=True)
    shutil.copy2(args.config, results_folder + '/config.yaml')

    if not args.do_not_use_wandb:
        wandb.save(results_folder + '/config.yaml')

    if cfg.DATASET.DATASET == 'h36m':
        if cfg.DATASET.DATA_FORMAT == 'h5':
            train_dataset = H36MDatasetH5(
                cfg.DATASET.ROOT + cfg.DATASET.TRAINFILE, train_set=True,
                include_heatmap=cfg.MODEL.CONDITION_KEY=="heatmaps"
            )
        else:
            raise RuntimeError("Incorrect data format choice for dataset, available [h5, pickle], attempted: {}".format(cfg.DATASET.DATA_FORMAT))
    elif cfg.DATASET.DATASET == '3dpw':
        train_dataset = PW3DDatasetH5(
            cfg.DATASET.ROOT + cfg.DATASET.TRAINFILE, train_set=True,
            include_heatmap=cfg.MODEL.CONDITION_KEY=='heatmaps'
        )
    else:
        raise RuntimeError("Incorrect dataset, available for training: [h36m, 3dpw], attempted: {}".format(
            cfg.DATASET.DATASET))


    trainer = Trainer(
        diffusion,
        train_dataset,
        train_batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        train_lr=cfg.TRAIN.LR,
        train_num_steps=cfg.TRAIN.NUM_STEPS,                                # total training steps
        gradient_accumulate_every=cfg.TRAIN.GRADIENT_ACCUMULATE_EVERY,      # gradient accumulation steps
        ema_decay=cfg.TRAIN.EMA_DECAY,                                      # exponential moving average decay
        amp=cfg.TRAIN.AMP,                                                  # turn on mixed precision
        condition_type=cfg.MODEL.CONDITION_TYPE,
        prior_type=cfg.MODEL.DIFFUSION_PRIOR,
        results_folder=results_folder,
        config=cfg,
        use_wandb=not args.do_not_use_wandb,
        random_seed=args.seed
    )

    # Train the model, save the weights and perform evaluation
    trainer.train()
    trainer.save('final')
    del train_dataset

    if cfg.DATASET.DATASET != '3dpw':
        """Test Human 3.6m full"""
        trainer.test(quick_eval_stride=16, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU)
        """Test Human 3.6m hard"""
        trainer.test_hard(batch_size=cfg.TEST.BATCH_SIZE_PER_GPU)
    else:
        """Test 3DPW"""
        trainer.test(quick_eval_stride=1, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU)

