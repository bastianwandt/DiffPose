import torch
import torch.nn as nn
import numpy as np

from torch.utils import data
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms.functional import to_pil_image

from pathlib import Path
from torch.optim import Adam, AdamW
from sklearn.metrics import auc

from tqdm.auto import tqdm
from ema_pytorch import EMA
from einops import rearrange, repeat

from utils.eval_functions import (compute_CP_list, pa_hypo_batch,
                                  err_3dpe_parallel, compute_3DPCK,
                                  calculate_ece, calculate_symmetry_error,
                                  sc_hypo_batch, get_scale, apply_scale)
from utils.data_utils import (
    reinsert_root_joint_torch, root_center_poses, mpii_to_h36m, mpii_to_h36m_covar, H36M_NAMES, MPII_NAMES
)
from data.dataset import H36MDatasetH5, HPDatasetH5, PW3DDatasetH5
from models.modules.utils import cycle, num_to_groups
from viz.pose_plotly import get_plotly_pose_fig, get_multiple_plotly_pose_figs
from viz.plot_p3d import plot17j_multi, plot17j
from models.modules.utils import extract, default
from typing import Literal


_CONDITION = Literal["pose_2d"]


class Trainer(object):
    def __init__(
        self,
        diffusion_model: nn.Module,
        dataset: data.dataset,
        *,
        ema_decay: float = 0.995,
        train_batch_size: int = 32,
        train_lr: float = 1e-4,
        train_num_steps: int = 100000,
        gradient_accumulate_every: int = 2,
        amp: bool = False,
        step_start_ema: int = 2000,
        ema_update_every: int = 10,
        save_and_sample_every: int = 10000,
        results_folder: str = './results',
        condition_type: _CONDITION = 'pose_2d',
        config: object = None,  # yacs.config.CfgNode
        use_wandb: bool = True,
        use_step_lr: bool = False,
        use_exp_decay: bool = False,
        random_seed: int = None
    ):
        super().__init__()
        self.image_size = diffusion_model.image_size
        self.config = config
        self.use_wandb = use_wandb

        self.model = diffusion_model
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.clip_grad = config.TRAIN.CLIP_GRAD

        # Conditioning key definition
        self.condition_type = condition_type
        self.cond_key = config.MODEL.CONDITION_KEY

        if dataset is not None:
            self.ds = dataset
            import random
            if random_seed is not None:
                def seed_worker(worker_id):
                    worker_seed = torch.initial_seed() % 2**32
                    np.random.seed(worker_seed)
                    random.seed(worker_seed)

                g = torch.Generator()
                g.manual_seed(random_seed)

                self.dl = cycle(data.DataLoader(dataset, batch_size=32, shuffle=True, generator=g))
            else:
                self.dl = cycle(data.DataLoader(dataset, batch_size=32, shuffle=True))
            #self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 1))

        if self.config.TRAIN.OPTIMIZER == 'adam':
            self.opt = Adam(diffusion_model.parameters(), lr = train_lr)
        else:
            self.opt = AdamW(diffusion_model.parameters(), lr = train_lr)  # Includes weight decay

        self.use_step_lr = use_step_lr
        self.use_exp_decay = use_exp_decay
        if self.use_step_lr:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.opt, step_size=int(train_num_steps*2/3), gamma=0.1
            )
        elif self.use_exp_decay:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.opt, gamma=0.995
            )
        else:
            self.scheduler = None


        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        if self.use_wandb:
            import wandb
            wandb.save(str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema.load_state_dict(data['ema'])
        self.scaler.load_state_dict(data['scaler'])

    def train(self):
        self.model.train()
        print_tqdm_interval = 10000
        with tqdm(initial = self.step, total = self.train_num_steps, miniters=print_tqdm_interval, mininterval=60) as pbar:

            while self.step < self.train_num_steps:
                step_loss = 0.
                for i in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    poses_3d = data['poses_3d']
                    cond = data[self.cond_key]

                    with autocast(enabled=self.amp):
                        loss = self.model(poses_3d, cond)
                        self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                    if self.step % print_tqdm_interval == 0:
                        pbar.set_description(f'loss: {loss.item():.4f}')
                    step_loss += loss.item() / self.gradient_accumulate_every

                self.scaler.step(self.opt)
                self.scaler.update()
                if self.scheduler is not None:
                    if self.use_step_lr:
                        self.scheduler.step()
                    else:
                        if self.step % 1000:
                            self.scheduler.step()
                self.opt.zero_grad()

                self.ema.update()

                if self.use_wandb:
                    import wandb
                    wandb.log({'loss': step_loss}, step=self.step)

                # if True:
                if self.step != 0 and self.step % self.save_and_sample_every == 0 and self.use_wandb:
                    self.ema.ema_model.eval()
                    poses_3d_z0 = self.ema.ema_model.sample(
                        cond, batch_size=cond.shape[0], sample_mean_only=True
                    )
                    poses_3d_z0 = rearrange(poses_3d_z0, '(b 1) (d j) -> b (d j)', d=3)

                    poses_3d_z0 = reinsert_root_joint_torch(poses_3d_z0, njoints=self.config.DATASET.NUM_JOINTS)
                    poses_3d_z0 = root_center_poses(poses_3d_z0, njoints=self.config.DATASET.NUM_JOINTS) * 1000

                    x_gt = data['p3d_gt']

                    total_err_z0_p1 = torch.mean(torch.mean(torch.sqrt(torch.sum((x_gt.reshape(cond.shape[0], 3, self.config.DATASET.NUM_JOINTS+1)
                                                                                  - poses_3d_z0.reshape(cond.shape[0], 3, self.config.DATASET.NUM_JOINTS+1)) ** 2,
                                                                                 dim=1)), dim=1)).item()
                    x_cpu = x_gt.cpu()
                    poses_3d_z0 = poses_3d_z0.cpu()

                    # protocol II
                    total_err_z0_p2 = err_3dpe_parallel(x_cpu, poses_3d_z0, return_sum=False, njoints=self.config.DATASET.NUM_JOINTS).mean().item()

                    # Plotting
                    poses_3d_z0 = rearrange(poses_3d_z0, 'b (d j) -> b d j', d=3, j=self.config.DATASET.NUM_JOINTS+1).cpu().numpy()
                    poses_3d_gt = rearrange(x_gt, 'b d j -> b d j', d=3, j=self.config.DATASET.NUM_JOINTS+1).cpu().numpy()

                    """Visualization code using wandb"""
                    for pose_idx in range(0, poses_3d_z0.shape[0]-1, 10):
                        p3d_z0 = poses_3d_z0[None, pose_idx]
                        p3d_gt = poses_3d_z0[pose_idx]

                        if self.config.DATASET.NUM_JOINTS != 16:
                            index_tensor = [0, 2, 5, 8, 1, 4, 7, 3, 6, 12, 15, 16, 18, 20, 17, 19, 21]
                            p3d_z0 = p3d_z0.take(index_tensor, axis=-1)
                            p3d_gt = p3d_gt.take(index_tensor, axis=-1)

                        fig = get_multiple_plotly_pose_figs(
                            p3d_z0,
                            p3d_gt
                        )

                        plot_name = "Train pose {}".format(str(pose_idx))

                        wandb.log({
                            plot_name: fig,
                        }, step=self.step)


                    wandb.log({
                        'MPJPE': total_err_z0_p1,
                        'PA-MPJPE': total_err_z0_p2
                    }, step=self.step)

                self.step += 1
                pbar.update(1)

        print('training complete')

    @torch.no_grad()
    def test(self, quick_eval_stride=16, batch_size=64, use_hard_set=False, use_validation=False, n_hypo=200, use_ema_model=False):
        self.model.eval()
        self.ema.ema_model.eval()

        cps_min_th = 1
        cps_max_th = 300
        cps_step = 1
        cps_length = int((cps_max_th + 1 - cps_min_th) / cps_step)

        # Protocol-I
        final_errors_z0_p1 = []
        final_errors_mean_p1 = []
        final_errors_best_p1 = []
        final_errors_worst_p1 = []
        final_errors_median_p1 = []

        # Protocol-I Scale Corrected
        final_errors_z0_p1sc = []
        final_errors_best_p1sc = []
        final_errors_pck_p1sc = []

        # Protocol-II
        final_errors_z0_p2 = []
        final_errors_mean_p2 = []
        final_errors_best_p2 = []
        final_errors_worst_p2 = []
        final_errors_median_p2 = []

        final_hypo_stddev = torch.zeros((3, self.config.DATASET.NUM_JOINTS+1))

        std_dev = 1.0

        if self.config.DATASET.DATASET == 'h36m':
            actions = [
                'Directions', 'Discussion', 'Eating', 'Greeting',
                'Phoning', 'Photo', 'Posing', 'Purchases','Sitting',
                'SittingDown', 'Smoking', 'Waiting', 'WalkDog',
                'WalkTogether', 'Walking'
            ]
        elif self.config.DATASET.DATASET == '3DHP':
            actions = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']
        elif self.config.DATASET.DATASET == '3dpw':
            actions = ['te']

        if use_hard_set:
            fn_extra = "_hardset"
            quick_eval = False
        else:
            fn_extra = ""
            quick_eval = True

        fn_extra = fn_extra + "_nhypo_{}_".format(n_hypo)

        fn_extra += self.config.DATASET.DATASET

        f = open(self.results_folder.joinpath("eval_" + self.condition_type + fn_extra + ".txt").__str__(), 'w')
        f.write("Evaluated on every %d-th frame with %d different hypotheses\nand standard dev of %.2f.\n\n\n" %
                (quick_eval_stride * 4, n_hypo, std_dev))
        print("Evaluated on every %d-th frame with %d different hypotheses\nand standard dev of %.2f.\n\n\n" %
                (quick_eval_stride * 4, n_hypo, std_dev))

        def eval_hypo_stddev(poses_3d, hypo_dim=1, batch_dim=0):
            # poses_3d.shape == (n_hypo, bs, 3, 17)
            # compute var over hypos and sum over poses for correct mean estimation over all poses in dataset
            return torch.sum(torch.std(poses_3d, dim=hypo_dim), dim=batch_dim).cpu()

        for action_idx, action in enumerate(actions):
            if self.config.DATASET.DATA_FORMAT == 'h5':
                if use_validation:
                    dataset_name = self.config.DATASET.VALFILE
                else:
                    dataset_name = self.config.DATASET.TESTFILE
                if self.config.DATASET.DATASET == 'h36m':
                    test_dataset = H36MDatasetH5(
                        self.config.DATASET.ROOT + dataset_name, train_set=False,
                        quick_eval=quick_eval, quick_eval_stride=quick_eval_stride,
                        actions=[action], hardsubset=use_hard_set, use_validation=use_validation
                    )
                elif self.config.DATASET.DATASET == '3DHP':
                    test_dataset = HPDatasetH5(
                        self.config.DATASET.ROOT + dataset_name, train_set=False,
                        quick_eval=False, quick_eval_stride=quick_eval_stride,
                        actions=[action], hardsubset=use_hard_set, use_validation=use_validation
                    )
                elif self.config.DATASET.DATASET == '3dpw':
                    test_dataset = PW3DDatasetH5(
                        self.config.DATASET.ROOT + dataset_name, train_set=False,
                        quick_eval=False, quick_eval_stride=quick_eval_stride,
                        actions=[action], hardsubset=use_hard_set, use_validation=use_validation
                    )
            else:
                raise RuntimeError(
                    "Incorrect data format choice for dataset, available [h5, pickle], attempted: {}".format(
                        self.config.DATASET.DATA_FORMAT))

            print('{} - {} contains {} samples'.format(self.config.DATASET.DATASET, action, len(test_dataset)))
            loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

            n_poses = len(test_dataset)

            total_err_z0_p1 = 0
            total_err_mean_p1 = 0
            total_err_worst_p1 = 0
            total_err_best_p1 = 0
            total_err_median_p1 = 0

            # Protocol-I-SC (Scale corrected)
            total_err_z0_p1sc = 0
            total_err_best_p1sc = 0
            total_err_mean_p1sc = 0
            total_best_pck_oracle_p1sc = 0

            total_err_z0_p2 = 0
            total_err_mean_p2 = 0
            total_err_worst_p2 = 0
            total_err_best_p2 = 0
            total_err_median_p2 = 0

            total_symmetry_error_z0 = 0
            total_symmetry_error = 0
            total_best_pck_oracle_p1 = 0
            total_auc_cps_p2_best = torch.zeros((cps_length,))

            predicted_poses_z0 = []
            predicted_poses_sampled = []
            ground_truth_poses = []

            hypo_stddev = torch.zeros((3, self.config.DATASET.NUM_JOINTS+1))

            pbar = tqdm(loader, total=len(loader))
            for batch_idx, sample in enumerate(pbar):
                pbar.set_description("Action: {}".format(action))

                x = sample['poses_3d']
                y_gt = sample[self.cond_key]
                bs = x.shape[0]

                if use_ema_model:
                    poses_3d_z0 = self.ema.ema_model.sample(
                        y_gt, batch_size=y_gt.shape[0], sample_mean_only=True
                    )
                else:
                    poses_3d_z0 = self.model.sample(
                        y_gt, batch_size=y_gt.shape[0], sample_mean_only=True
                    )
                poses_3d_z0 = rearrange(poses_3d_z0, '(b 1) (d j) -> b (d j)', d=3)

                poses_3d_z0 = reinsert_root_joint_torch(poses_3d_z0, njoints=self.config.DATASET.NUM_JOINTS)
                poses_3d_z0 = root_center_poses(poses_3d_z0, njoints=self.config.DATASET.NUM_JOINTS) * 1000
                x_gt = sample['p3d_gt']

                total_symmetry_error_z0 += calculate_symmetry_error(poses_3d_z0.reshape(bs, 3, self.config.DATASET.NUM_JOINTS+1), reduction='sum')

                predicted_poses_z0.append(poses_3d_z0.reshape(bs, 3, self.config.DATASET.NUM_JOINTS+1).cpu().numpy())
                ground_truth_poses.append(x_gt.reshape(bs, 3, self.config.DATASET.NUM_JOINTS+1).cpu().numpy())

                total_err_z0_p1 += torch.sum(torch.mean(torch.sqrt(torch.sum((x_gt.reshape(bs, 3, self.config.DATASET.NUM_JOINTS+1)
                                                                              - poses_3d_z0.reshape(bs, 3, self.config.DATASET.NUM_JOINTS+1)) ** 2,
                                                                             dim=1)), dim=1)).item()
                x_cpu = x_gt.cpu()
                poses_3d_z0 = poses_3d_z0.cpu()

                # Protocol I-SC
                gt_scale = get_scale(x_cpu.unsqueeze(1))
                poses_3d_z0_sc = apply_scale(rearrange(poses_3d_z0, 'b (d j) -> b 1 d j', d=3), gt_scale)
                total_err_z0_p1sc = torch.sum(torch.mean(torch.sqrt(torch.sum((x_cpu.reshape(bs, 3, self.config.DATASET.NUM_JOINTS+1)
                                                                     - poses_3d_z0_sc.reshape(bs, 3, self.config.DATASET.NUM_JOINTS+1)
                                                                     ) ** 2, dim=-2)), dim=-1)).item()

                # protocol II
                total_err_z0_p2 += err_3dpe_parallel(x_cpu, poses_3d_z0, njoints=self.config.DATASET.NUM_JOINTS)

                if use_ema_model:
                    poses_3d_pred = self.ema.ema_model.sample(
                        y_gt, batch_size=y_gt.shape[0], n_hypotheses_to_sample=n_hypo
                    )
                else:
                    poses_3d_pred = self.model.sample(
                        y_gt, batch_size=y_gt.shape[0], n_hypotheses_to_sample=n_hypo
                    )

                poses_3d_pred = rearrange(poses_3d_pred, '(b p) (d j) -> (b p) (d j)', p=n_hypo, d=3)

                poses_3d_pred = reinsert_root_joint_torch(poses_3d_pred, njoints=self.config.DATASET.NUM_JOINTS)
                poses_3d_pred = root_center_poses(poses_3d_pred, njoints=self.config.DATASET.NUM_JOINTS) * 1000
                poses_3d_pred = rearrange(poses_3d_pred, '(b p) (d j) -> b p d j', b=bs, p=n_hypo, d=3, j=self.config.DATASET.NUM_JOINTS+1)

                total_symmetry_error += torch.sum(
                    calculate_symmetry_error(poses_3d_pred.reshape(bs, n_hypo, 3, self.config.DATASET.NUM_JOINTS+1),
                                             reduction='mean', njoints=self.config.DATASET.NUM_JOINTS)
                )

                predicted_poses_sampled.append(poses_3d_pred.cpu().numpy())

                # compute variance in x, y and z direction
                hypo_stddev += eval_hypo_stddev(poses_3d_pred)
                errors_proto1 = torch.mean(torch.sqrt(torch.sum((x_gt.reshape(bs, 1, 3, self.config.DATASET.NUM_JOINTS+1)
                                                                 - poses_3d_pred) ** 2, dim=2)), dim=2)

                errors_pck_p1 = compute_3DPCK(x_gt.reshape(bs, 1, 3, self.config.DATASET.NUM_JOINTS+1), poses_3d_pred)

                # procrustes is faster on cpu
                poses_3d_pred = poses_3d_pred.cpu()
                x_gt = x_gt.cpu()
                x_gt = repeat(x_gt, 'b d j -> b n_hypo d j', n_hypo=n_hypo, d=3, j=self.config.DATASET.NUM_JOINTS+1)

                poses_3d_pred_sc = apply_scale(poses_3d_pred, gt_scale)

                errors_proto1_sc = torch.mean(torch.sqrt(torch.sum((x_gt - poses_3d_pred_sc) ** 2, dim=-2)), dim=-1)
                errors_pck_p1_sc = compute_3DPCK(x_gt, poses_3d_pred_sc)

                errors_proto2 = err_3dpe_parallel(x_gt, poses_3d_pred.clone(), return_sum=False, njoints=self.config.DATASET.NUM_JOINTS).reshape(bs, -1)
                poses_3d_pa = pa_hypo_batch(x_gt, poses_3d_pred, njoints=self.config.DATASET.NUM_JOINTS).reshape(bs, n_hypo, 3, self.config.DATASET.NUM_JOINTS+1)

                x_gt = x_gt.reshape(bs, n_hypo, 3, self.config.DATASET.NUM_JOINTS+1)[:, 0]

                errors_auc_cps_p2 = compute_CP_list(
                    x_gt.reshape(bs, 1, 3, self.config.DATASET.NUM_JOINTS+1).cuda(), poses_3d_pa.cuda(),
                    min_th=cps_min_th, max_th=cps_max_th, step=cps_step
                )

                # finished evaluating a single batch, need to compute hypo statistics per gt pose!
                # best hypos
                values, _ = torch.min(errors_proto1, dim=1)
                total_err_best_p1 += torch.sum(values).item()

                values, _ = torch.min(errors_proto1_sc, dim=1)
                total_err_best_p1sc += torch.sum(values).item()

                total_err_mean_p1 += torch.sum(torch.mean(errors_proto1, dim=1))
                total_err_mean_p1sc += torch.sum(torch.mean(errors_proto1_sc, dim=1))
                total_err_mean_p2 += torch.sum(torch.mean(errors_proto2, dim=1))

                # best pck hypos
                values, _ = torch.max(errors_pck_p1, dim=1)
                total_best_pck_oracle_p1 += torch.sum(values).item()

                values, _ = torch.max(errors_pck_p1_sc, dim=1)
                total_best_pck_oracle_p1sc += torch.sum(values).item()

                # best auc cps hypose
                values, _ = torch.max(errors_auc_cps_p2, dim=1)
                total_auc_cps_p2_best += torch.sum(values, dim=0)

                # worst hypos
                values, _ = torch.max(errors_proto1, dim=1)
                total_err_worst_p1 += torch.sum(values).item()

                # median hypos
                values, _ = torch.median(errors_proto1, dim=1)
                total_err_median_p1 += torch.sum(values).item()
                # Protocol-II:
                # best hypos
                values, _ = torch.min(errors_proto2, dim=1)
                total_err_best_p2 += torch.sum(values).item()

                # worst hypos
                values, _ = torch.max(errors_proto2, dim=1)
                total_err_worst_p2 += torch.sum(values).item()

                # median hypos
                values, _ = torch.median(errors_proto2, dim=1)
                total_err_median_p2 += torch.sum(values).item()

            # Calculate CPS
            # from list of cp values (one element per threshold), compute AUC CPS:
            k_list = np.arange(cps_min_th, cps_max_th + 1, cps_step)
            total_auc_cps_p2_best /= n_poses
            cps_auc_p2_best = auc(k_list, total_auc_cps_p2_best.cpu().numpy())

            # write result for single action to file:
            f.write("Action: %s\n" % action)
            f.write("3D Protocol-I z_0: %.2f\n" % (total_err_z0_p1 / n_poses))
            f.write("3D Protocol-I best hypo: %.2f\n" % (total_err_best_p1 / n_poses))
            f.write("3D Protocol-I median hypo: %.2f\n" % (total_err_median_p1 / n_poses))
            f.write("3D Protocol-I mean hypo: %.2f\n" % (total_err_mean_p1 / n_poses))
            f.write("3D Protocol-I worst hypo: %.2f\n" % (total_err_worst_p1 / n_poses))

            f.write("3D PCK best: %.2f\n" % (100.*total_best_pck_oracle_p1 / n_poses))
            f.write("3D CPS best: %.2f\n" % cps_auc_p2_best)
            f.write("Symmetry error z0 (mm): %.4f\n" % (total_symmetry_error_z0 / n_poses))
            f.write("Symmetry error (mm): %.4f\n" % (total_symmetry_error / n_poses))
            f.write("\n\n")

            f.write("3D Protocol-I-SC best hypo: %.2f\n" % (total_err_best_p1sc / n_poses))
            f.write("3D Protocol-I-SC z_0: %.2f\n" % (total_err_z0_p1sc / n_poses))
            f.write("3D PCK-SC best: %.2f\n" % (100.*total_best_pck_oracle_p1sc / n_poses))
            f.write("\n\n")

            f.write("3D Protocol-II z_0: %.2f\n" % (total_err_z0_p2 / n_poses))
            f.write("3D Protocol-II best hypo: %.2f\n" % (total_err_best_p2 / n_poses))
            f.write("3D Protocol-II median hypo: %.2f\n" % (total_err_median_p2 / n_poses))
            f.write("3D Protocol-II mean hypo: %.2f\n" % (total_err_mean_p2 / n_poses))
            f.write("3D Protocol-II worst hypo: %.2f\n" % (total_err_worst_p2 / n_poses))
            f.write("\n\n")
            final_errors_z0_p1.append(total_err_z0_p1 / n_poses)
            final_errors_mean_p1.append(total_err_mean_p1 / n_poses)
            final_errors_best_p1.append(total_err_best_p1 / n_poses)
            final_errors_worst_p1.append(total_err_worst_p1 / n_poses)
            final_errors_median_p1.append(total_err_median_p1 / n_poses)

            final_errors_z0_p1sc.append(total_err_z0_p1sc / n_poses)
            final_errors_best_p1sc.append(total_err_best_p1sc / n_poses)
            final_errors_pck_p1sc.append(total_best_pck_oracle_p1sc / n_poses)

            final_errors_z0_p2.append(total_err_z0_p2 / n_poses)
            final_errors_mean_p2.append(total_err_mean_p2 / n_poses)
            final_errors_best_p2.append(total_err_best_p2 / n_poses)
            final_errors_worst_p2.append(total_err_worst_p2 / n_poses)
            final_errors_median_p2.append(total_err_median_p2 / n_poses)

            final_hypo_stddev += (hypo_stddev / n_poses)

            if self.use_wandb:
                import wandb

            if self.use_wandb:
                ground_truth_poses = np.concatenate(ground_truth_poses, axis=0)
                predicted_poses_z0 = np.concatenate(predicted_poses_z0, axis=0)
                predicted_poses_sampled = np.concatenate(predicted_poses_sampled, axis=0)

                if self.config.DATASET.NUM_JOINTS != 16:
                    index_tensor = [0, 2, 5, 8, 1, 4, 7, 3, 6, 12, 15, 16, 18, 20, 17, 19, 21]
                    ground_truth_poses = ground_truth_poses.take(index_tensor, axis=-1)
                    predicted_poses_z0 = predicted_poses_z0.take(index_tensor, axis=-1)
                    predicted_poses_sampled = predicted_poses_sampled.take(index_tensor, axis=-1)

                for pose_idx in range(0, ground_truth_poses.shape[0] - 1, 64):
                    fig = get_multiple_plotly_pose_figs(
                        predicted_poses_z0[None, pose_idx],
                        ground_truth_poses[pose_idx]
                    )
                    sampled_fig = get_multiple_plotly_pose_figs(
                        predicted_poses_sampled[pose_idx, ::10]
                    )

                    if not use_hard_set:
                        plot_name = "{} pose {}".format(action, str(pose_idx))
                        sampled_plot_name = "{} sampled poses {}".format(action, str(pose_idx))
                    else:
                        plot_name = "{} pose {} (hard set)".format(action, str(pose_idx))
                        sampled_plot_name = "{} sampled poses {} (hard set)".format(action, str(pose_idx))

                    wandb.log({
                        plot_name: fig,
                        sampled_plot_name: sampled_fig
                    })

                    wandb.save(f.name)

        avg_z0_p1 = sum(final_errors_z0_p1) / len(final_errors_z0_p1)
        avg_mean_p1 = sum(final_errors_mean_p1) / len(final_errors_mean_p1)
        avg_best_p1 = sum(final_errors_best_p1) / len(final_errors_best_p1)
        avg_worst_p1 = sum(final_errors_worst_p1) / len(final_errors_worst_p1)
        avg_median_p1 = sum(final_errors_median_p1) / len(final_errors_median_p1)

        avg_z0_p1sc = sum(final_errors_z0_p1sc) / len(final_errors_z0_p1sc)
        avg_best_p1sc = sum(final_errors_best_p1sc) / len(final_errors_best_p1sc)
        avg_pck_p1sc = sum(final_errors_pck_p1sc) / len(final_errors_pck_p1sc)

        avg_z0_p2 = sum(final_errors_z0_p2) / len(final_errors_z0_p2)
        avg_mean_p2 = sum(final_errors_mean_p2) / len(final_errors_mean_p2)
        avg_best_p2 = sum(final_errors_best_p2) / len(final_errors_best_p2)
        avg_worst_p2 = sum(final_errors_worst_p2) / len(final_errors_worst_p2)
        avg_median_p2 = sum(final_errors_median_p2) / len(final_errors_median_p2)

        # results averaged over all actions
        f.write("Average: \n")
        f.write("3D Protocol-I z_0: %.2f\n" % avg_z0_p1)
        f.write("3D Protocol-I best hypo: %.2f\n" % avg_best_p1)
        f.write("3D Protocol-I median hypo: %.2f\n" % avg_median_p1)
        f.write("3D Protocol-I mean hypo: %.2f\n" % avg_mean_p1)
        f.write("3D Protocol-I worst hypo: %.2f\n" % avg_worst_p1)

        f.write("3D Protocol-I SC best hypo: %.2f\n" % avg_best_p1sc)
        f.write("3D Protocol-I SC z0: %.2f\n" % avg_z0_p1sc)
        f.write("3D Protocol-I SC PCK: %.2f\n" % (avg_pck_p1sc * 100.))

        f.write("3D Protocol-II z_0: %.2f\n" % avg_z0_p2)
        f.write("3D Protocol-II best hypo: %.2f\n" % avg_best_p2)
        f.write("3D Protocol-II median hypo: %.2f\n" % avg_median_p2)
        f.write("3D Protocol-II mean hypo: %.2f\n" % avg_mean_p2)
        f.write("3D Protocol-II worst hypo: %.2f\n" % avg_worst_p2)

        if use_hard_set:
            print("\n Results on hard subset")
        else:
            print("\n Results on full set")
        print("\nAverage:")
        print("3D Protocol-I z_0: %.2f" % avg_z0_p1)
        print("3D Protocol-I best hypo: %.2f" % avg_best_p1)
        print("3D Protocol-I median hypo: %.2f" % avg_median_p1)
        print("3D Protocol-I mean hypo: %.2f" % avg_mean_p1)
        print("3D Protocol-I worst hypo: %.2f\n" % avg_worst_p1)

        print("3D Protocol-I SC best hypo: %.2f\n" % avg_best_p1sc)
        print("3D Protocol-I SC z0: %.2f\n" % avg_z0_p1sc)
        print("3D Protocol-I SC PCK: %.2f\n" % (avg_pck_p1sc * 100.))

        print("3D Protocol-II z_0: %.2f" % avg_z0_p2)
        print("3D Protocol-II best hypo: %.2f" % avg_best_p2)
        print("3D Protocol-II median hypo: %.2f" % avg_median_p2)
        print("3D Protocol-II mean hypo: %.2f" % avg_mean_p2)
        print("3D Protocol-II worst hypo: %.2f" % avg_worst_p2)

        std_dev_in_mm = final_hypo_stddev / len(actions)
        # standard deviation in mm per dimension and per joint:
        print("\nstd dev per joint and dim in mm:")
        f.write("\n\n")
        f.write("std dev per joint and dim in mm:\n")
        for i in range(std_dev_in_mm.shape[1]):
            print("joint %d: std_x=%.2f, std_y=%.2f, std_z=%.2f" % (i, std_dev_in_mm[0, i], std_dev_in_mm[1, i],
                                                                    std_dev_in_mm[2, i]))
            f.write("joint %d: std_x=%.2f, std_y=%.2f, std_z=%.2f\n" % (i, std_dev_in_mm[0, i], std_dev_in_mm[1, i],
                                                                        std_dev_in_mm[2, i]))

        std_dev_means = torch.mean(std_dev_in_mm, dim=1)
        print("mean: std_x=%.2f, std_y=%.2f, std_z=%.2f" % (std_dev_means[0], std_dev_means[1], std_dev_means[2]))
        f.write("mean: std_x=%.2f, std_y=%.2f, std_z=%.2f\n" % (std_dev_means[0], std_dev_means[1], std_dev_means[2]))

        if self.use_wandb:
            if use_hard_set:
                extra_name = "(hard) "
            else:
                extra_name = ""
            # Log the metrics
            wandb.run.summary["MPJPE " + extra_name + "- z0"] = avg_z0_p1
            wandb.run.summary["MPJPE " + extra_name + "- best"] = avg_best_p1
            wandb.run.summary["MPJPE " + extra_name + "- mean"] = avg_mean_p1

            wandb.run.summary["PA-MPJPE " + extra_name + "- z0"] = avg_z0_p2
            wandb.run.summary["PA-MPJPE " + extra_name + "- best"] = avg_best_p2
            wandb.run.summary["PA-MPJPE " + extra_name + "- mean"] = avg_mean_p2

            # Log the joint variances
            # standard deviation in mm per dimension and per joint:
            for i in range(std_dev_in_mm.shape[1]):
                wandb.run.summary["Joint " + extra_name + "{}: std x".format(i)] = std_dev_in_mm[0, i]
                wandb.run.summary["Joint " + extra_name + "{}: std y".format(i)] = std_dev_in_mm[1, i]
                wandb.run.summary["Joint " + extra_name + "{}: std z".format(i)] = std_dev_in_mm[2, i]

            wandb.run.summary["Mean joint std" + extra_name + ": x"] = std_dev_means[0]
            wandb.run.summary["Mean joint std" + extra_name + ": y"] = std_dev_means[1]
            wandb.run.summary["Mean joint std" + extra_name + ": z"] = std_dev_means[2]

    @torch.no_grad()
    def test_hard(self, batch_size=64, n_hypo=200, use_ema_model=False):
        self.ema.ema_model.eval()
        self.model.eval()

        std_dev = 1.0

        cps_min_th = 1
        cps_max_th = 300
        cps_step = 1
        cps_length = int((cps_max_th + 1 - cps_min_th) / cps_step)

        # Create a wandb table to store some samples and heatmaps
        if self.use_wandb:
            if self.config.DATASET.NUM_JOINTS == 16:
                columns = ["id", "Pose_GT", "Pose_Z0", "Sampled_poses", "MPJPE_best", "PA-MPJPE_best"]
            else:
                columns = ["id", "MPJPE_best", "PA-MPJPE_best"]
            for i in range(self.config.DATASET.COND_JOINTS):
                columns.append("Heatmap_{}".format(MPII_NAMES[i]))

            import wandb
            table_hard = wandb.Table(
                columns=columns
            )

        fn_extra = "_hardset_nhypo_{}".format(n_hypo)
        quick_eval = False

        f = open(self.results_folder.joinpath("full_eval_" + self.condition_type + fn_extra + ".txt").__str__(), 'w')
        f.write("Evaluated on every %d-th frame with %d different hypotheses\nand standard dev of %.2f.\n\n\n" %
                (4, n_hypo, std_dev))
        print("Evaluated on every %d-th frame with %d different hypotheses\nand standard dev of %.2f.\n\n\n" %
                (4, n_hypo, std_dev))

        def eval_hypo_stddev(poses_3d, hypo_dim=1, batch_dim=0):
            # poses_3d.shape == (n_hypo, bs, 3, 17)
            # compute var over hypos and sum over poses for correct mean estimation over all poses in dataset
            return torch.sum(torch.std(poses_3d, dim=hypo_dim), dim=batch_dim).cpu()

        if self.config.DATASET.DATA_FORMAT == 'h5':
            test_dataset = H36MDatasetH5(
                self.config.DATASET.ROOT + self.config.DATASET.TESTFILE, train_set=False,
                quick_eval=False,
                hardsubset=True
            )
        else:
            raise RuntimeError(
                "Incorrect data format choice for dataset, available [h5, pickle], attempted: {}".format(
                    self.config.DATASET.DATA_FORMAT))

        print('{} - hardset contains {} samples'.format(self.config.DATASET.DATASET, len(test_dataset)))

        loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        n_poses = len(test_dataset)

        worst_err_best_p1 = 0
        worst_err_best_p2 = 0

        total_err_z0_p1 = 0
        total_err_meanshift_p1 = 0
        total_err_mean_p1 = 0
        total_err_worst_p1 = 0
        total_err_best_p1 = 0
        total_err_median_p1 = 0

        total_err_z0_p2 = 0
        total_err_meanshift_p2 = 0
        total_err_mean_p2 = 0
        total_err_worst_p2 = 0
        total_err_best_p2 = 0
        total_err_median_p2 = 0

        total_symmetry_error_z0 = 0
        total_symmetry_error = 0

        total_ece_sampled = []
        predicted_poses_z0 = []
        predicted_poses_sampled = []
        ground_truth_poses = []

        hypo_stddev = torch.zeros((3, self.config.DATASET.NUM_JOINTS+1))
        total_best_pck_oracle_p1 = 0
        total_auc_cps_p2_best = torch.zeros((cps_length,))

        pbar = tqdm(loader, total=len(loader))
        for batch_idx, sample in enumerate(pbar):
            pbar.set_description("Hardset:")

            x = sample['poses_3d']
            y_gt = sample[self.cond_key]
            bs = x.shape[0]

            if use_ema_model:
                poses_3d_z0 = self.ema.ema_model.sample(
                    y_gt, batch_size=y_gt.shape[0], sample_mean_only=True
                )
            else:
                poses_3d_z0 = self.model.sample(
                    y_gt, batch_size=y_gt.shape[0], sample_mean_only=True
                )
            poses_3d_z0 = rearrange(poses_3d_z0, '(b 1) (d j) -> b (d j)', d=3, j=self.config.DATASET.NUM_JOINTS)

            poses_3d_z0 = reinsert_root_joint_torch(poses_3d_z0, njoints=self.config.DATASET.NUM_JOINTS)
            poses_3d_z0 = root_center_poses(poses_3d_z0, njoints=self.config.DATASET.NUM_JOINTS) * 1000
            x_gt = sample['p3d_gt']

            total_symmetry_error_z0 += calculate_symmetry_error(poses_3d_z0.reshape(bs, 3, self.config.DATASET.NUM_JOINTS+1), reduction='sum')

            predicted_poses_z0.append(poses_3d_z0.reshape(bs, 3, self.config.DATASET.NUM_JOINTS+1).cpu().numpy())
            ground_truth_poses.append(x_gt.reshape(bs, 3, self.config.DATASET.NUM_JOINTS+1).cpu().numpy())

            total_err_z0_p1 += torch.sum(torch.mean(torch.sqrt(torch.sum((x_gt.reshape(bs, 3, self.config.DATASET.NUM_JOINTS+1)
                                                                          - poses_3d_z0.reshape(bs, 3, self.config.DATASET.NUM_JOINTS+1)) ** 2,
                                                                         dim=1)), dim=1)).item()
            x_cpu = x_gt.cpu()
            poses_3d_z0 = poses_3d_z0.cpu()

            # protocol II
            total_err_z0_p2 += err_3dpe_parallel(x_cpu, poses_3d_z0, njoints=self.config.DATASET.NUM_JOINTS)

            if use_ema_model:
                poses_3d_pred = self.ema.ema_model.sample(
                    y_gt, batch_size=y_gt.shape[0], n_hypotheses_to_sample=n_hypo
                )
            else:
                poses_3d_pred = self.model.sample(
                    y_gt, batch_size=y_gt.shape[0], n_hypotheses_to_sample=n_hypo
                )

            poses_3d_pred = rearrange(poses_3d_pred, '(b p) (d j) -> (b p) (d j)', p=n_hypo, d=3)
            poses_3d_pred = reinsert_root_joint_torch(poses_3d_pred, njoints=self.config.DATASET.NUM_JOINTS)
            poses_3d_pred = root_center_poses(poses_3d_pred, njoints=self.config.DATASET.NUM_JOINTS) * 1000
            poses_3d_pred = rearrange(poses_3d_pred, '(b p) (d j) -> b p d j', b=bs, p=n_hypo, d=3, j=self.config.DATASET.NUM_JOINTS+1)


            total_symmetry_error += torch.sum(calculate_symmetry_error(poses_3d_pred.reshape(bs, n_hypo, 3, self.config.DATASET.NUM_JOINTS+1), reduction='mean'))
            total_ece_sampled.append(calculate_ece(x_gt.reshape(bs, 3, self.config.DATASET.NUM_JOINTS+1), poses_3d_pred))
            predicted_poses_sampled.append(poses_3d_pred.cpu().numpy())

            # compute variance in x, y and z direction
            hypo_stddev += eval_hypo_stddev(poses_3d_pred)

            errors_proto1 = torch.mean(torch.sqrt(torch.sum((x_gt.reshape(bs, 1, 3, self.config.DATASET.NUM_JOINTS+1)
                                                             - poses_3d_pred) ** 2, dim=2)), dim=2)

            errors_pck_p1 = compute_3DPCK(x_gt.reshape(bs, 1, 3, self.config.DATASET.NUM_JOINTS+1), poses_3d_pred)

            # procrustes is faster on cpu
            poses_3d_pred = poses_3d_pred.cpu()
            x_gt = x_gt.cpu()
            x_gt = repeat(x_gt, 'b d j -> b n_hypo d j', n_hypo=n_hypo, d=3, j=self.config.DATASET.NUM_JOINTS+1)

            errors_proto2 = err_3dpe_parallel(x_gt, poses_3d_pred.clone(), return_sum=False, njoints=self.config.DATASET.NUM_JOINTS).reshape(bs, -1)
            poses_3d_pa = pa_hypo_batch(x_gt, poses_3d_pred, njoints=self.config.DATASET.NUM_JOINTS).reshape(bs, n_hypo, 3, self.config.DATASET.NUM_JOINTS+1)

            x_gt = x_gt.reshape(bs, n_hypo, 3, self.config.DATASET.NUM_JOINTS+1)[:, 0]

            errors_auc_cps_p2 = compute_CP_list(
                x_gt.reshape(bs, 1, 3, self.config.DATASET.NUM_JOINTS+1).cuda(), poses_3d_pa.cuda(),
                min_th=cps_min_th, max_th=cps_max_th, step=cps_step
            )

            # finished evaluating a single batch, need to compute hypo statistics per gt pose!
            # best hypos
            values, _ = torch.min(errors_proto1, dim=1)
            total_err_best_p1 += torch.sum(values).item()
            worst_err_best_p1 = max(worst_err_best_p1, values.max().item())

            total_err_mean_p1 += torch.sum(torch.mean(errors_proto1, dim=1))
            total_err_mean_p2 += torch.sum(torch.mean(errors_proto2, dim=1))

            # best pck hypos
            values, _ = torch.max(errors_pck_p1, dim=1)
            total_best_pck_oracle_p1 += torch.sum(values).item()

            # best auc cps hypose
            values, _ = torch.max(errors_auc_cps_p2, dim=1)
            total_auc_cps_p2_best += torch.sum(values, dim=0)

            # worst hypos
            values, _ = torch.max(errors_proto1, dim=1)
            total_err_worst_p1 += torch.sum(values).item()

            # median hypos
            values, _ = torch.median(errors_proto1, dim=1)
            total_err_median_p1 += torch.sum(values).item()

            # Protocol-II: (PA-MPJPE)
            # best hypos
            values, _ = torch.min(errors_proto2, dim=1)
            total_err_best_p2 += torch.sum(values).item()
            worst_err_best_p2 = max(worst_err_best_p2, values.max().item())

            # worst hypos
            values, _ = torch.max(errors_proto2, dim=1)
            total_err_worst_p2 += torch.sum(values).item()

            # median hypos
            values, _ = torch.median(errors_proto2, dim=1)
            total_err_median_p2 += torch.sum(values).item()

            # Log data to wandb table
            if self.use_wandb:
                pose_idx = batch_idx*bs
                ["id", "Pose_GT_Z0", "Sampled_poses", "MPJPE", "PA-MPJPE"]

                if self.config.DATASET.NUM_JOINTS == 16:
                    gt_fig = plot17j(ground_truth_poses[-1][0], return_fig=True)
                    z0_fig = plot17j(predicted_poses_z0[-1][0], return_fig=True)
                    if predicted_poses_sampled[-1].shape[1] > 20:
                        sampled_figs = plot17j_multi(
                            rearrange(predicted_poses_sampled[-1][0, :20], 'p d j -> p (d j)', d=3, j=self.config.DATASET.NUM_JOINTS+1).reshape(-1, 3*(self.config.DATASET.NUM_JOINTS+1)),
                            return_fig=True
                        )
                    else:
                        sampled_figs = plot17j_multi(
                            rearrange(predicted_poses_sampled[-1][0, :], 'p d j -> p (d j)', d=3, j=self.config.DATASET.NUM_JOINTS+1).reshape(-1, 3*(self.config.DATASET.NUM_JOINTS+1)),
                            return_fig=True
                        )


                    data_row = [
                        pose_idx,
                        wandb.Image(gt_fig),
                        wandb.Image(z0_fig),
                        wandb.Image(sampled_figs),
                        torch.min(errors_proto1[0]).item(),
                        torch.min(errors_proto2[0]).item(),
                    ]
                    import matplotlib.pyplot as plt
                    # TODO: Check if this actually does close them (It complains about too many figures open without it)
                    plt.close(gt_fig)
                    plt.close(z0_fig)
                    plt.close(sampled_figs)
                    plt.close()
                else:
                    data_row = [
                        pose_idx,
                        torch.min(errors_proto1[0]).item(),
                        torch.min(errors_proto2[0]).item(),
                    ]


                for i in range(self.config.DATASET.COND_JOINTS):
                    data_row.append(wandb.Image(to_pil_image(
                        sample["heatmaps"][0, i].clamp(min=0, max=1)
                    )))

                table_hard.add_data(*data_row)


        if self.use_wandb:
            wandb.log({"Table_hard": table_hard})

        # Calculate ECE
        total_ece_sampled = torch.cat(total_ece_sampled, dim=0)
        ece_metric, _ = total_ece_sampled.mean(dim=0).median(dim=-1)

        # Calculate CPS
        # from list of cp values (one element per threshold), compute AUC CPS:
        k_list = np.arange(cps_min_th, cps_max_th + 1, cps_step)
        total_auc_cps_p2_best /= n_poses
        cps_auc_p2_best = auc(k_list, total_auc_cps_p2_best.cpu().numpy())

        # write result for single action to file:
        f.write("Hardset: \n")
        f.write("3D Protocol-I z_0: %.2f\n" % (total_err_z0_p1 / n_poses))
        f.write("3D Protocol-I meanshift: %.2f\n" % (total_err_meanshift_p1 / n_poses))
        f.write("3D Protocol-I best hypo: %.2f\n" % (total_err_best_p1 / n_poses))
        f.write("3D Protocol-I median hypo: %.2f\n" % (total_err_median_p1 / n_poses))
        f.write("3D Protocol-I mean hypo: %.2f\n" % (total_err_mean_p1 / n_poses))
        f.write("3D Protocol-I worst hypo: %.2f\n" % (total_err_worst_p1 / n_poses))
        f.write("3D Protocol-I worst err for best hypo: %.2f\n" % worst_err_best_p1)

        f.write("3D Protocol-II z_0: %.2f\n" % (total_err_z0_p2 / n_poses))
        f.write("3D Protocol-II meanshift: %.2f\n" % (total_err_meanshift_p2 / n_poses))
        f.write("3D Protocol-II best hypo: %.2f\n" % (total_err_best_p2 / n_poses))
        f.write("3D Protocol-II median hypo: %.2f\n" % (total_err_median_p2 / n_poses))
        f.write("3D Protocol-II mean hypo: %.2f\n" % (total_err_mean_p2 / n_poses))
        f.write("3D Protocol-II worst hypo: %.2f\n" % (total_err_worst_p2 / n_poses))
        f.write("3D Protocol-II worst err for best hypo: %.2f\n" % worst_err_best_p2)

        f.write("3D PCK best: %.2f\n" % (100.*total_best_pck_oracle_p1 / n_poses))
        f.write("3D CPS best: %.2f\n" % cps_auc_p2_best)

        f.write("Symmetry error z0 (mm): %.4f\n" % (total_symmetry_error_z0 / n_poses))
        f.write("Symmetry error (mm): %.4f\n" % (total_symmetry_error / n_poses))

        f.write("\n\n")

        print("Hardset: \n")
        print("3D Protocol-I z_0: %.2f\n" % (total_err_z0_p1 / n_poses))
        print("3D Protocol-I meanshift: %.2f\n" % (total_err_meanshift_p1 / n_poses))
        print("3D Protocol-I best hypo: %.2f\n" % (total_err_best_p1 / n_poses))
        print("3D Protocol-I median hypo: %.2f\n" % (total_err_median_p1 / n_poses))
        print("3D Protocol-I mean hypo: %.2f\n" % (total_err_mean_p1 / n_poses))
        print("3D Protocol-I worst hypo: %.2f\n" % (total_err_worst_p1 / n_poses))
        print("3D Protocol-I worst err for best hypo: %.2f\n" % worst_err_best_p1)

        print("3D Protocol-II z_0: %.2f\n" % (total_err_z0_p2 / n_poses))
        print("3D Protocol-II meanshift: %.2f\n" % (total_err_meanshift_p2 / n_poses))
        print("3D Protocol-II best hypo: %.2f\n" % (total_err_best_p2 / n_poses))
        print("3D Protocol-II median hypo: %.2f\n" % (total_err_median_p2 / n_poses))
        print("3D Protocol-II mean hypo: %.2f\n" % (total_err_mean_p2 / n_poses))
        print("3D Protocol-II worst hypo: %.2f\n" % (total_err_worst_p2 / n_poses))
        print("3D Protocol-II worst err for best hypo: %.2f\n" % worst_err_best_p2)


        print("3D PCK best: %.2f\n" % (100.*total_best_pck_oracle_p1 / n_poses))
        print("3D CPS best: %.2f\n" % cps_auc_p2_best)
        print("Symmetry error z0 (mm): %.4f\n" % (total_symmetry_error_z0 / n_poses))
        print("Symmetry error (mm): %.4f\n" % (total_symmetry_error / n_poses))

        print("\n\n")

        if self.use_wandb and self.config.DATASET.NUM_JOINTS == 16:
            import wandb
            ground_truth_poses = np.concatenate(ground_truth_poses, axis=0)
            predicted_poses_z0 = np.concatenate(predicted_poses_z0, axis=0)
            predicted_poses_sampled = np.concatenate(predicted_poses_sampled, axis=0)

            for pose_idx in range(0, ground_truth_poses.shape[0] - 1, 64):
                fig = get_multiple_plotly_pose_figs(
                    predicted_poses_z0[None, pose_idx],
                    ground_truth_poses[pose_idx]
                )
                sampled_fig = get_multiple_plotly_pose_figs(
                    predicted_poses_sampled[pose_idx, ::10]
                )

                plot_name = "Pose {} (hard set)".format(str(pose_idx))
                sampled_plot_name = "Sampled poses {} (hard set)".format(str(pose_idx))

                wandb.log({
                    plot_name: fig,
                    sampled_plot_name: sampled_fig
                })

                wandb.save(f.name)

        std_dev_in_mm = hypo_stddev / n_poses
        # standard deviation in mm per dimension and per joint:
        print("\nstd dev per joint and dim in mm:")
        f.write("\n\n")
        f.write("std dev per joint and dim in mm:\n")
        for i in range(std_dev_in_mm.shape[1]):
            print("joint %d: std_x=%.2f, std_y=%.2f, std_z=%.2f" % (i, std_dev_in_mm[0, i], std_dev_in_mm[1, i],
                                                                    std_dev_in_mm[2, i]))
            f.write("joint %d: std_x=%.2f, std_y=%.2f, std_z=%.2f\n" % (i, std_dev_in_mm[0, i], std_dev_in_mm[1, i],
                                                                        std_dev_in_mm[2, i]))

        std_dev_means = torch.mean(std_dev_in_mm, dim=1)
        print("mean: std_x=%.2f, std_y=%.2f, std_z=%.2f" % (std_dev_means[0], std_dev_means[1], std_dev_means[2]))
        f.write("mean: std_x=%.2f, std_y=%.2f, std_z=%.2f\n" % (std_dev_means[0], std_dev_means[1], std_dev_means[2]))

        if self.use_wandb:
            extra_name = "(hard) "
            # Log the metrics
            wandb.run.summary["MPJPE " + extra_name + "- z0"] = total_err_z0_p1 / n_poses
            wandb.run.summary["MPJPE " + extra_name + "- best"] = total_err_best_p1 / n_poses
            wandb.run.summary["MPJPE " + extra_name + "- mean"] = total_err_mean_p1 / n_poses

            wandb.run.summary["PA-MPJPE " + extra_name + "- z0"] = total_err_z0_p2 / n_poses
            wandb.run.summary["PA-MPJPE " + extra_name + "- best"] = total_err_best_p2 / n_poses
            wandb.run.summary["PA-MPJPE " + extra_name + "- mean"] = total_err_mean_p2 / n_poses

            wandb.run.summary["CPS " + extra_name] = cps_auc_p2_best
            wandb.run.summary["PCK " + extra_name] = 100.*total_best_pck_oracle_p1 / n_poses
            wandb.run.summary["ECE " + extra_name] = ece_metric
            wandb.run.summary["Sym z0 " + extra_name] = (total_symmetry_error_z0 / n_poses)
            wandb.run.summary["Sym " + extra_name] = (total_symmetry_error / n_poses)

            # Log the joint variances
            # standard deviation in mm per dimension and per joint:
            for i in range(std_dev_in_mm.shape[1]):
                wandb.run.summary["Joint " + extra_name + "{}: std x".format(i)] = std_dev_in_mm[0, i]
                wandb.run.summary["Joint " + extra_name + "{}: std y".format(i)] = std_dev_in_mm[1, i]
                wandb.run.summary["Joint " + extra_name + "{}: std z".format(i)] = std_dev_in_mm[2, i]

            wandb.run.summary["Mean joint std" + extra_name + ": x"] = std_dev_means[0]
            wandb.run.summary["Mean joint std" + extra_name + ": y"] = std_dev_means[1]
            wandb.run.summary["Mean joint std" + extra_name + ": z"] = std_dev_means[2]


    @torch.no_grad()
    def predict(self, sample, n_hypo=200, use_ema_model=False):
        self.ema.ema_model.eval()
        self.model.eval()

        x = sample['poses_3d']
        y_gt = sample[self.cond_key]

        bs = x.shape[0]
        if use_ema_model:
            poses_3d_z0 = self.ema.ema_model.sample(
                y_gt, batch_size=y_gt.shape[0], sample_mean_only=True
            )
        else:
            poses_3d_z0 = self.model.sample(
                y_gt, batch_size=y_gt.shape[0], sample_mean_only=True
            )
        poses_3d_z0 = rearrange(poses_3d_z0, '(b 1) (d j) -> b (d j)', d=3)

        poses_3d_z0 = reinsert_root_joint_torch(poses_3d_z0, njoints=self.config.DATASET.NUM_JOINTS)
        poses_3d_z0 = root_center_poses(poses_3d_z0, njoints=self.config.DATASET.NUM_JOINTS) * 1000
        poses_3d_z0 = rearrange(poses_3d_z0, 'b (d j) -> b d j', d=3, j=self.config.DATASET.NUM_JOINTS+1)
        x_gt = sample['p3d_gt']

        x_cpu = x_gt.cpu()
        poses_3d_z0 = poses_3d_z0.cpu()

        # sample multiple z
        if use_ema_model:
            poses_3d_pred = self.ema.ema_model.sample(
                y_gt, batch_size=y_gt.shape[0], n_hypotheses_to_sample=n_hypo
            )
        else:
            poses_3d_pred = self.model.sample(
                y_gt, batch_size=y_gt.shape[0], n_hypotheses_to_sample=n_hypo
            )
        poses_3d_pred = rearrange(poses_3d_pred, '(b p) (d j) -> (b p) (d j)', p=n_hypo, d=3, j=self.config.DATASET.NUM_JOINTS)
        poses_3d_pred = reinsert_root_joint_torch(poses_3d_pred, njoints=self.config.DATASET.NUM_JOINTS)
        poses_3d_pred = root_center_poses(poses_3d_pred) * 1000
        poses_3d_pred = rearrange(poses_3d_pred, '(b p) (d j) -> b p d j', d=3, j=self.config.DATASET.NUM_JOINTS+1, p=n_hypo)

        x_cpu = rearrange(x_cpu, 'b d j -> b d j', d=3, j=self.config.DATASET.NUM_JOINTS+1)

        return poses_3d_z0.cpu(), poses_3d_pred.cpu(), x_cpu
