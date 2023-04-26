# Code from: https://github.com/twehrbein/Probabilistic-Monocular-3D-Human-Pose-Estimation-with-Normalizing-Flows/blob/main/data/data_h36m.py
import torch
from torch.utils.data import Dataset
import config as c
import numpy as np
from utils.data_utils import normalize_poses, preprocess_gaussian_fits
import pickle
import tables as tb
from torchvision import transforms
from typing import NamedTuple
from einops import rearrange


settings = {
    'H36M': {
        'subjects_train': ['S1', 'S5', 'S6', 'S7', 'S8'],
        'subjects_test': ['S9', 'S11'],
        'subjects_train_wo_val': ['S1', 'S6', 'S7'],
        'subjects_val': ['S5', 'S8'],

        'actions_all': ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases',
               'Sitting', 'SittingDown', 'Smoking', 'Waiting', 'WalkDog', 'WalkTogether', 'Walking'],
        'subactions_all': ['0', '1'],
        'cameras_all': ['54138969', '55011271', '58860488', '60457274']
    },
    '3DHP': {
        'subjects_to_use': ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']
    }
}


class SampleH5(NamedTuple):
    poses_3d: torch.Tensor
    p3d_gt: torch.Tensor

    p2d_hrnet: torch.Tensor
    p2d_hrnet_unnorm: torch.Tensor
    p2d_gt: torch.Tensor

    heatmap: torch.Tensor


class PW3DDatasetH5(Dataset):
    """
    3DPW dataset stored as hdf5
    """
    def __init__(self, h5_file, quick_eval=False, quick_eval_stride=16,
                 train_set=True, actions=None, hardsubset=False,
                 device='cuda', include_heatmap=True,
                 use_validation=False
                 ):

        num_joints_3d = 23
        num_joints_2d = 18
        num_cond_joints = 16
        self.device = device
        data = tb.open_file(h5_file, mode="r")

        if train_set:
            self.datasplit = data.root.dataset.trainset
            subjects_to_use = ['tr'] #, 'va']  # Do not train on validation
        else:
            self.datasplit = data.root.dataset.testset
            subjects_to_use = ['te']

        poses_3d = []
        p2d_hrnet_unnorm = []
        p2d_gt = []
        heatmaps = []

        if quick_eval:
            print("")
            print("###################################")
            print("Warning: quick eval is not available for 3DPW, setting it to False")
            print("###################################")
            print("")
            quick_eval = False

        stride = 1
        if quick_eval:
            stride = quick_eval_stride

        for i, sample in enumerate(self.datasplit.iterrows(stop=None, step=stride)):
            if sample['subject'].decode('utf-8') not in subjects_to_use:
                continue
            poses_3d.append(1000 * sample['gt_3d'])
            p2d_gt.append(sample['gt_2d'])
            p2d_hrnet_unnorm.append(sample['argmax_2d'])

            # To further decrease the required memory and improve speed
            if not include_heatmap:
                heatmaps.append(torch.randn(1))
            else:
                heatmaps.append(torch.from_numpy(sample['heatmap']))

        poses_3d = np.stack(poses_3d, axis=0)
        p2d_hrnet_unnorm = np.stack(p2d_hrnet_unnorm, axis=0)
        p2d_hrnet_unnorm = rearrange(p2d_hrnet_unnorm, 'b d j -> b (d j)')
        p2d_gt = np.stack(p2d_gt, axis=0)

        # preprocess 3d gt poses
        p3d_gt = rearrange(poses_3d.copy(), 'b j d -> b d j', d=3)
        # root center gt poses
        p3d_gt -= p3d_gt[:, :, 0, None]
        # invert y and z axes:
        p3d_gt[:, 1:, :] *= -1
        # Don't remove root joint
        poses_3d = rearrange(poses_3d, 'b j d -> b d j', d=3)
        # poses_3d = poses_3d[:, :, 1:]
        poses_3d = rearrange(poses_3d, 'b d j -> b (d j)')

        # preprocess 3d gt poses
        p2d_gt = rearrange(p2d_gt.copy(), 'b d j -> b d j', d=3)
        p2d_gt = p2d_gt[:, :-1, :]  # Remove joint confidence from joint position
        # root center gt poses
        p2d_gt -= p2d_gt[:, :, 0, None]
        # invert y axes:
        p2d_gt[:, 1:, :] *= -1
        p2d_gt = rearrange(p2d_gt, 'b d j -> b (d j)')

        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        data.close()
        print(poses_3d.shape)
        poses_3d = normalize_poses(poses_3d, njoints=num_joints_3d+1)
        p2d_hrnet = normalize_poses(p2d_hrnet_unnorm.copy(), njoints=num_cond_joints)
        # joints_to_use_2d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]
        # Remove the thorax which doesn't exist in the MPII skeleton in order to make the gt and the heatmaps match
        p2d_gt = rearrange(normalize_poses(p2d_gt, njoints=num_joints_2d), 'b (d j) -> b (j d)', d=2)

        self.dataset = []
        for sample in zip(
            heatmaps,
            p3d_gt, poses_3d,
            p2d_hrnet_unnorm, p2d_hrnet, p2d_gt
        ):
            self.dataset.append(
                SampleH5(
                    heatmap=sample[0],
                    p3d_gt=torch.from_numpy(sample[1]),
                    poses_3d=torch.from_numpy(sample[2]),
                    p2d_hrnet_unnorm=torch.from_numpy(sample[3]),
                    p2d_hrnet=torch.from_numpy(sample[4]),
                    p2d_gt=torch.from_numpy(sample[5])
                )
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the corresponding row from the data table
        sample = self.dataset[idx]
        heatmap = sample.heatmap.float()

        sample = {
            'poses_3d': sample.poses_3d,
            'p2d_hrnet': sample.p2d_hrnet,
            'p3d_gt': sample.p3d_gt,
            'p2d_hrnet_unnorm': sample.p2d_hrnet_unnorm,
            'p2d_gt': sample.p2d_gt,
            'heatmaps': heatmap,
        }

        for key in sample:
            if sample[key] is None:
                continue
            sample[key] = sample[key].to('cuda')

        return sample


class HPDatasetH5(Dataset):
    """
    MPI-INF-3DHP Dataset (https://vcai.mpi-inf.mpg.de/3dhp-dataset/)
    """
    def __init__(self, h5_file, quick_eval=False, quick_eval_stride=16,
                 train_set=True, actions=None, hardsubset=False,
                 device='cuda', include_heatmap=True,
                 use_validation=False
                 ):

        self.device = device
        data = tb.open_file(h5_file, mode="r")

        self.datasplit = data.root.dataset.testset
        assert actions is None
        actions = settings['3DHP']['subjects_to_use']

        poses_3d = []
        p2d_hrnet_unnorm = []
        p2d_gt = []
        heatmaps = []

        if quick_eval:
            print("")
            print("###################################")
            print("Warning: quick eval is not available for 3DHP, setting it to False")
            print("###################################")
            print("")
            quick_eval = False

        stride = 1
        if quick_eval:
            stride = quick_eval_stride

        for i, sample in enumerate(self.datasplit.iterrows(stop=None, step=stride)):
            if sample['action'].decode('utf-8') not in actions:
                continue

            poses_3d.append(sample['gt_3d_uni'])
            p2d_gt.append(sample['gt_2d'])
            p2d_hrnet_unnorm.append(sample['argmax_2d'])

            # To further decrease the required memory and improve speed
            if not include_heatmap:
                heatmaps.append(torch.randn(1))
            else:
                heatmaps.append(torch.from_numpy(sample['heatmap']))

        poses_3d = np.stack(poses_3d, axis=0)
        p2d_hrnet_unnorm = np.stack(p2d_hrnet_unnorm, axis=0)
        p2d_gt = np.stack(p2d_gt, axis=0)

        # preprocess 3d gt poses
        p3d_gt = rearrange(poses_3d.copy(), 'b (d j) -> b d j', d=3)
        # root center gt poses
        p3d_gt -= p3d_gt[:, :, 0, None]
        # invert y and z axes:
        p3d_gt[:, 1:, :] *= -1
        # remove root joint
        poses_3d = rearrange(poses_3d, 'b (d j) -> b d j', d=3)
        poses_3d = poses_3d[:, :, 1:]
        poses_3d = rearrange(poses_3d, 'b d j -> b (d j)')

        # preprocess 3d gt poses
        p2d_gt = rearrange(p2d_gt.copy(), 'b (d j) -> b d j', d=2)
        # root center gt poses
        p2d_gt -= p2d_gt[:, :, 0, None]
        # invert y axes:
        p2d_gt[:, 1:, :] *= -1
        p2d_gt = rearrange(p2d_gt, 'b d j -> b (d j)')

        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        data.close()
        poses_3d = normalize_poses(poses_3d)
        p2d_hrnet = normalize_poses(p2d_hrnet_unnorm.copy())
        joints_to_use_2d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]
        # Remove the thorax which doesn't exist in the MPII skeleton in order to make the gt and the heatmaps match
        p2d_gt = rearrange(p2d_gt, 'b (d j) -> b d j', d=2)[:, :, joints_to_use_2d]
        p2d_gt = rearrange(p2d_gt, 'b d j -> b (d j)', d=2)
        p2d_gt = rearrange(normalize_poses(p2d_gt), 'b (d j) -> b (j d)', d=2)

        self.dataset = []
        for sample in zip(
            heatmaps,
            p3d_gt, poses_3d,
            p2d_hrnet_unnorm, p2d_hrnet, p2d_gt
        ):
            self.dataset.append(
                SampleH5(
                    heatmap = sample[0],
                    p3d_gt = torch.from_numpy(sample[1]),
                    poses_3d = torch.from_numpy(sample[2]),
                    p2d_hrnet_unnorm = torch.from_numpy(sample[3]),
                    p2d_hrnet = torch.from_numpy(sample[4]),
                    p2d_gt = torch.from_numpy(sample[5])
                )
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the corresponding row from the data table
        sample = self.dataset[idx]
        heatmap = sample.heatmap.float()

        sample = {
            'poses_3d': sample.poses_3d,
            'p2d_hrnet': sample.p2d_hrnet,
            'p3d_gt': sample.p3d_gt,
            'p2d_hrnet_unnorm': sample.p2d_hrnet_unnorm,
            'p2d_gt': sample.p2d_gt,
            'heatmaps': heatmap,
        }

        for key in sample:
            if sample[key] is None:
                continue
            sample[key] = sample[key].to('cuda')

        return sample


class H36MDatasetH5(Dataset):
    """
    Human 3.6m dataset (full and hard subset) stored as hdf5
    """
    def __init__(self, h5_file, quick_eval=False, quick_eval_stride=16,
                 train_set=True, actions=settings['H36M']['actions_all'], hardsubset=False,
                 device='cuda', include_heatmap=True,
                 use_validation=False
                 ):

        self.device = device
        data = tb.open_file(h5_file, mode="r")

        if use_validation:
            self.datasplit = data.root.dataset.trainset
            if train_set:
                subjects_to_use = settings['H36M']['subjects_train_wo_val']
            else:
                subjects_to_use = settings['H36M']['subjects_val']
        else:
            if train_set:
                self.datasplit = data.root.dataset.trainset
                subjects_to_use = settings['H36M']['subjects_train']
            else:
                self.datasplit = data.root.dataset.testset
                subjects_to_use = settings['H36M']['subjects_test']

        poses_3d = []
        p2d_hrnet_unnorm = []
        p2d_gt = []
        heatmaps = []

        if quick_eval and hardsubset:
            print("")
            print("###################################")
            print("Warning: quick eval should not be combined with hardsubset for the final evaluation")
            print("###################################")
            print("")

        stride = 1
        if quick_eval:
            stride = quick_eval_stride

        self.frame_info =[]

        for i, sample in enumerate(self.datasplit.iterrows(stop=None, step=stride)):
            if sample['action'].decode('utf-8') not in actions:
                continue
            if sample['subject'].decode('utf-8') not in subjects_to_use:
                continue

            # Filter out hard subset
            if hardsubset:
                if 'hardsubset' in self.datasplit.coldescrs.keys():
                    if not sample['hardsubset']:
                        continue
                else:
                    print('Hardsubset is not available')
                    return

            self.frame_info.append([sample['action'], sample['subject'], sample['subaction'], sample['cam'], sample['frame_orig']])
            poses_3d.append(sample['gt_3d'])
            p2d_gt.append(sample['gt_2d'])
            p2d_hrnet_unnorm.append(sample['argmax_2d'])

            # To further decrease the required memory and improve speed
            if not include_heatmap:
                heatmaps.append(torch.randn(1))
            else:
                heatmaps.append(torch.from_numpy(sample['heatmap']))

        poses_3d = np.stack(poses_3d, axis=0)
        p2d_hrnet_unnorm = np.stack(p2d_hrnet_unnorm, axis=0)
        p2d_gt = np.stack(p2d_gt, axis=0)

        # preprocess 3d gt poses
        p3d_gt = rearrange(poses_3d.copy(), 'b d j -> b d j', d=3)
        # root center gt poses
        p3d_gt -= p3d_gt[:, :, 0, None]
        # invert y and z axes:
        p3d_gt[:, 1:, :] *= -1
        # remove root joint
        poses_3d = rearrange(poses_3d, 'b d j -> b d j', d=3)
        poses_3d = poses_3d[:, :, 1:]
        poses_3d = rearrange(poses_3d, 'b d j -> b (d j)')

        # preprocess 3d gt poses
        p2d_gt = rearrange(p2d_gt.copy(), 'b d j -> b d j', d=2)
        # root center gt poses
        p2d_gt -= p2d_gt[:, :, 0, None]
        # invert y axes:
        p2d_gt[:, 1:, :] *= -1
        p2d_gt = rearrange(p2d_gt, 'b d j -> b (d j)')

        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        data.close()
        poses_3d = normalize_poses(poses_3d)
        p2d_hrnet = normalize_poses(rearrange(p2d_hrnet_unnorm.copy(), 'b d j -> b (d j)'))
        joints_to_use_2d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]
        # Remove the thorax which doesn't exist in the MPII skeleton in order to make the gt and the heatmaps match
        p2d_gt = rearrange(p2d_gt, 'b (d j) -> b d j', d=2)[:, :, joints_to_use_2d]
        p2d_gt = rearrange(p2d_gt, 'b d j -> b (d j)', d=2)
        p2d_gt = rearrange(normalize_poses(p2d_gt), 'b (d j) -> b (j d)', d=2)

        self.dataset = []
        for sample in zip(
            heatmaps,
            p3d_gt, poses_3d,
            p2d_hrnet_unnorm, p2d_hrnet, p2d_gt
        ):
            self.dataset.append(
                SampleH5(
                    heatmap = sample[0],
                    p3d_gt = torch.from_numpy(sample[1]),
                    poses_3d = torch.from_numpy(sample[2]),
                    p2d_hrnet_unnorm = torch.from_numpy(sample[3]),
                    p2d_hrnet = torch.from_numpy(sample[4]),
                    p2d_gt = torch.from_numpy(sample[5])
                )
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the corresponding row from the data table
        sample = self.dataset[idx]
        heatmap = sample.heatmap.float()

        sample = {
            'poses_3d': sample.poses_3d,
            'p2d_hrnet': sample.p2d_hrnet,
            'p3d_gt': sample.p3d_gt,
            'p2d_hrnet_unnorm': sample.p2d_hrnet_unnorm,
            'p2d_gt': sample.p2d_gt,
            'heatmaps': heatmap,
        }

        for key in sample:
            if sample[key] is None:
                continue
            sample[key] = sample[key].to('cuda')

        return sample
