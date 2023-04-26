from pathlib import Path
import wandb

from models.trainer import Trainer
from models.diffusion import GaussianDiffusion
from models.denoiser import get_denoiser
from models.condition_embedding import get_condition_embedding
from experiments.default import get_config_file


if __name__ == '__main__':
    import argparse

    baseline_config = 'experiments/h36m/baseline.yaml'

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

    denoiser = get_denoiser(cfg)
    conditioner = get_condition_embedding(cfg)


    wandb.init(
        project="Eval",
        name=cfg.MODEL.NAME,
        entity="probabilistic_human_pose",
        config=wandb_config
    )

    num_joints = cfg.DATASET.NUM_JOINTS
    if num_joints != 16:
        num_joints += 1

    diffusion = GaussianDiffusion(
        denoiser,
        conditioner,
        image_size=128,
        objective=cfg.MODEL.DIFFUSION_OBJECTIVE,
        timesteps=cfg.MODEL.TIMESTEPS,   # number of steps
        loss_type=cfg.LOSS.LOSS_TYPE,    # L1 or L2
        scaling_3d_pose=cfg.TRAIN.SCALE_3D_POSE,  # Pre-conditioning of target scale
        noise_scale=cfg.MODEL.NOISE_SCALE,
        cosine_offset=cfg.TRAIN.COSINE_OFFSET,
        num_joints=num_joints
    ).cuda()

    run_name = wandb.run.name
    print("Run name: ", run_name, flush=True)
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
        use_wandb=args.do_not_use_wandb
    )

    trainer.load('final')
    if cfg.DATASET.DATASET == '3dpw':
        trainer.test(quick_eval_stride=1, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, n_hypo=cfg.TEST.NUM_HYPO)
    else:
        trainer.test(quick_eval_stride=16, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, n_hypo=cfg.TEST.NUM_HYPO)

    if cfg.DATASET.DATASET == 'h36m':
        trainer.test_hard(batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, n_hypo=cfg.TEST.NUM_HYPO)

    wandb.finish()
    print('done', flush=True)

