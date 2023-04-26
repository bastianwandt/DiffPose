import torch
from torchvision import transforms
import argparse
import data.pose_detection_2d.hrnet.model as hrnet
from yacs.config import CfgNode as CN
from PIL import Image
from experiments.default import get_config_file
from models.trainer import Trainer
from models.diffusion import GaussianDiffusion
from models.denoiser import get_denoiser
from models.condition_embedding import get_condition_embedding
from viz.plot_p3d import plot17j_multi, plot17j
from utils.data_utils import (
    reinsert_root_joint_torch, root_center_poses, mpii_to_h36m, mpii_to_h36m_covar, H36M_NAMES, MPII_NAMES
)
import numpy as np
import tables as tb
import imageio


def get_pretrained_model(
        config_file='data/pose_detection_2d/hrnet/mpii_hrnet_w32_255x255.yaml',
        # model_weights='../data/pose_detection_2d/fine_HRNet.pt'  # H36M finetuned
        model_weights='data/pose_detection_2d/pose_hrnet_w32_256x256.pth'  # Original MPII
):
    cfg = get_hrnet_config_file(config_file)
    m = hrnet.get_pose_net(
        cfg, False
    )

    model_weights = torch.load(model_weights)

    m.load_state_dict(model_weights)
    # m.load_state_dict(model_weights['net'])
    m.eval()

    return m


def get_hrnet_config_file(config_file):
    _C = CN()

    _C.OUTPUT_DIR = ''
    _C.LOG_DIR = ''
    _C.DATA_DIR = ''
    _C.GPUS = (0,)
    _C.WORKERS = 4
    _C.PRINT_FREQ = 20
    _C.AUTO_RESUME = False
    _C.PIN_MEMORY = True
    _C.RANK = 0

    # Cudnn related params
    _C.CUDNN = CN()
    _C.CUDNN.BENCHMARK = True
    _C.CUDNN.DETERMINISTIC = False
    _C.CUDNN.ENABLED = True

    # common params for NETWORK
    _C.MODEL = CN()
    _C.MODEL.NAME = 'pose_hrnet'
    _C.MODEL.INIT_WEIGHTS = True
    _C.MODEL.PRETRAINED = ''
    _C.MODEL.NUM_JOINTS = 17
    _C.MODEL.TAG_PER_JOINT = True
    _C.MODEL.TARGET_TYPE = 'gaussian'
    _C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
    _C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
    _C.MODEL.SIGMA = 2
    _C.MODEL.EXTRA = CN(new_allowed=True)

    _C.LOSS = CN()
    _C.LOSS.USE_OHKM = False
    _C.LOSS.TOPK = 8
    _C.LOSS.USE_TARGET_WEIGHT = True
    _C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

    # DATASET related params
    _C.DATASET = CN()
    _C.DATASET.ROOT = ''
    _C.DATASET.DATASET = 'mpii'
    _C.DATASET.TRAIN_SET = 'train'
    _C.DATASET.TEST_SET = 'valid'
    _C.DATASET.DATA_FORMAT = 'jpg'
    _C.DATASET.HYBRID_JOINTS_TYPE = ''
    _C.DATASET.SELECT_DATA = False

    # training data augmentation
    _C.DATASET.FLIP = True
    _C.DATASET.SCALE_FACTOR = 0.25
    _C.DATASET.ROT_FACTOR = 30
    _C.DATASET.PROB_HALF_BODY = 0.0
    _C.DATASET.NUM_JOINTS_HALF_BODY = 8
    _C.DATASET.COLOR_RGB = False

    # train
    _C.TRAIN = CN()

    _C.TRAIN.LR_FACTOR = 0.1
    _C.TRAIN.LR_STEP = [90, 110]
    _C.TRAIN.LR = 0.001

    _C.TRAIN.OPTIMIZER = 'adam'
    _C.TRAIN.MOMENTUM = 0.9
    _C.TRAIN.WD = 0.0001
    _C.TRAIN.NESTEROV = False
    _C.TRAIN.GAMMA1 = 0.99
    _C.TRAIN.GAMMA2 = 0.0

    _C.TRAIN.BEGIN_EPOCH = 0
    _C.TRAIN.END_EPOCH = 140

    _C.TRAIN.RESUME = False
    _C.TRAIN.CHECKPOINT = ''

    _C.TRAIN.BATCH_SIZE_PER_GPU = 32
    _C.TRAIN.SHUFFLE = True

    # testing
    _C.TEST = CN()

    # size of images for each device
    _C.TEST.BATCH_SIZE_PER_GPU = 32
    # Test Model Epoch
    _C.TEST.FLIP_TEST = False
    _C.TEST.POST_PROCESS = False
    _C.TEST.SHIFT_HEATMAP = False

    _C.TEST.USE_GT_BBOX = False

    # nms
    _C.TEST.IMAGE_THRE = 0.1
    _C.TEST.NMS_THRE = 0.6
    _C.TEST.SOFT_NMS = False
    _C.TEST.OKS_THRE = 0.5
    _C.TEST.IN_VIS_THRE = 0.0
    _C.TEST.COCO_BBOX_FILE = ''
    _C.TEST.BBOX_THRE = 1.0
    _C.TEST.MODEL_FILE = ''

    # debug
    _C.DEBUG = CN()
    _C.DEBUG.DEBUG = False
    _C.DEBUG.SAVE_BATCH_IMAGES_GT = False
    _C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
    _C.DEBUG.SAVE_HEATMAPS_GT = False
    _C.DEBUG.SAVE_HEATMAPS_PRED = False

    _C.defrost()
    _C.merge_from_file(config_file)
    _C.freeze()

    return _C


image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
        transforms.Resize([255, 255])
    ])


@torch.inference_mode()
def run_inference(data_dir, hrnet, diffpose, n_hypo=200):
    hrnet.eval()


    label_file = tb.open_file(data_dir + '/labels.h5')
    seqs = label_file.root.seq
    cams = label_file.root.cam
    frames = label_file.root.frame
    subjs = label_file.root.subj
    poses_3D = label_file.root['3D']
    poses_2D = label_file.root['2D']
    # poses_2D_px = 256*poses_2D

    for seq, cam, frame, subj, p3d in zip(seqs, cams, frames, subjs, poses_3D):
        seq, cam, frame, subj = int(seq), int(cam), int(frame), int(subj)

        image_name = data_dir + '/seq_{:03d}/cam_{:02d}/image_{:06d}.png'.format(seq, cam, frame)

        img = imageio.v2.imread(image_name)
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.show()

        heatmap = hrnet(image_transforms(img).unsqueeze(0))
        plt.imshow(heatmap[0].sum(0))
        plt.show()

        poses = diffpose.sample(heatmap, n_hypotheses_to_sample=n_hypo)
        print(poses.shape)

        exit()


if __name__ == '__main__':
    hrnet = get_pretrained_model()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='demo/config.yaml',
                        help='Path to experiment config file to use')
    parser.add_argument('--do_not_use_wandb', action='store_true', default=False,
                        help='Deactivates the use of wandb')
    parser.add_argument('--datadir', type=str, default='/media/karl/Samsung_T5/data/SkiPose/Ski-PosePTZ-CameraDataset-png/test',
                        help='Path to SkiPose dataset (test or train dir)')
    parser.add_argument('--model_weight', type=str, default='demo/model-final.pt',
                        help='Model weights of pretrained model')
    args = parser.parse_args()

    cfg = get_config_file(args.config)

    denoiser = get_denoiser(cfg)
    conditioner = get_condition_embedding(cfg)
    diffusion = GaussianDiffusion(
        denoiser,
        conditioner,
        image_size=128,
        objective=cfg.MODEL.DIFFUSION_OBJECTIVE,
        timesteps=cfg.MODEL.TIMESTEPS,  # number of steps
        loss_type=cfg.LOSS.LOSS_TYPE,  # L1 or L2
        scaling_3d_pose=cfg.TRAIN.SCALE_3D_POSE,  # Pre-conditioning of target scale
        noise_scale=cfg.MODEL.NOISE_SCALE,
        cosine_offset=cfg.TRAIN.COSINE_OFFSET
    ).cpu()

    diffusion.load_state_dict(torch.load(args.model_weight)['model'])
    diffusion.eval()

    run_inference(args.datadir, hrnet, diffusion)
