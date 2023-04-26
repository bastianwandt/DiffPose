import csv
from pathlib import Path
import imageio
import numpy as np
import matplotlib.pyplot as plt
import cdflib
import data.preprocessing.hrnet.model as hrnet
import torch
from yacs.config import CfgNode as CN
from torchvision import transforms
from viz.plot_p3d import plot17j_multi, plot17j
from utils.data_utils import H36M_NAMES, MPII_NAMES, H36M17j_TO_MPII, get_symmetry


def get_pretrained_model(
        config_file='data/pose_detection_2d/hrnet/mpii_hrnet_w32_255x255.yaml',
        use_h36m_model=True,
):
    if use_h36m_model:
        model_weights='data/pose_detection_2d/fine_HRNet.pt'  # H36M finetuned
    else:
        model_weights = 'data/pose_detection_2d/pose_hrnet_w32_256x256.pth'  # Original MPII

    cfg = get_hrnet_config_file(config_file)
    m = hrnet.get_pose_net(
        cfg, False
    )

    model_weights = torch.load(model_weights)

    if use_h36m_model:
        m.load_state_dict(model_weights['net'])
    else:
        m.load_state_dict(model_weights)
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


def get_crop_bb(joints, img_shape):
    # Estimate the crop based bounding box by finding the max/min joint coords and add a margin
    bb_coords = np.concatenate([
        joints.max(axis=0),
        joints.min(axis=0)
    ])

    print(bb_coords.shape, joints.shape)
    bb_center = np.round(0.5 * (bb_coords[0:2] + bb_coords[2:]))
    bb_width = np.abs(bb_coords[0] - bb_coords[2])
    bb_height = np.abs(bb_coords[1] - bb_coords[3])

    crop_size = np.round(max(bb_width, bb_height) * 1.2)

    crop_bb = np.round(np.concatenate([
        bb_center - 0.5 * crop_size,
        bb_center + 0.5 * crop_size
    ])).astype(np.int32)

    if crop_bb[0] < 0:
        crop_bb[2] -= crop_bb[0]
        crop_bb[0] -= crop_bb[0]

    if crop_bb[1] < 0:
        crop_bb[3] -= crop_bb[1]
        crop_bb[1] -= crop_bb[1]

    if crop_bb[3] > img_shape[0]:
        crop_bb[1] -= crop_bb[3]
        crop_bb[3] = img_shape[0]

    if crop_bb[2] > img_shape[1]:
        crop_bb[0] -= crop_bb[2]
        crop_bb[2] = img_shape[1]

    return crop_size, crop_bb


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


@torch.inference_mode()
def plot_qualitative_results_based_on_csv(
        csvfile, datadir, wehrbein, sharma, ours, gt
):
    poses_3d_wehrbein = np.load(wehrbein, allow_pickle=True)
    poses_3d_sharma = np.load(sharma, allow_pickle=True)['p3d_pred_sharma']
    poses_3d_sharma = np.swapaxes(poses_3d_sharma, 2, 3)
    poses_3d_sharma[:, :, -2] *= -1
    poses_3d_sharma[:, :, -1] *= -1
    poses_3d_ours = np.load(ours, allow_pickle=True)
    poses_3d_gt = np.load(gt, allow_pickle=True)

    net = get_pretrained_model(use_h36m_model=True)
    net.to('cuda')
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
        transforms.Resize([255, 255])
    ])
    # print(poses_3d_others.keys())
    # print(poses_3d_others.shape)

    with open(csvfile, 'r') as f:
        csvreader = csv.DictReader(f, delimiter=',')
        k=0
        for line in csvreader:
            if k < 2:
                k+=1
                continue
            fn = line['filename']
            frameidx = int(line['frameidx'])
            datasetidx = int(line['datasetidx'])
            hardsetidx = int(line['hardsetidx'])
            hardsetidx_alt = int(line['hardsetidx_alt'])
            img = imageio.get_reader(datadir + '/' + fn, 'ffmpeg').get_data(frameidx)
            img = np.asarray(img)

            p3d_wehrbein = poses_3d_wehrbein[:, hardsetidx_alt].reshape(-1, 3, 17) / 1000.
            p3d_sharma = poses_3d_sharma[:, hardsetidx_alt].reshape(-1, 3, 17) / 1000.
            print(poses_3d_ours.shape, poses_3d_gt.shape)
            p3d_ours = poses_3d_ours[hardsetidx].reshape(-1, 3, 17) / 1000.
            p3d_gt = poses_3d_gt[hardsetidx].reshape(-1, 3, 17).copy() / 1000.

            plot17j(p3d_gt.reshape(17*3))
            plot17j(p3d_ours[0].reshape(17*3))
            plot17j(p3d_sharma[0].reshape(17*3))
            plot17j(p3d_wehrbein[0].reshape(17*3))
            # gt = poses_3d_gt[hardsetidx_alt].reshape(-1, 17, 3)
            # gt[:, :, -2:] *= -1
            # pred = poses_3d_others[:, hardsetidx_alt].reshape(-1, 17, 3)
            # pred[:, :, -2:] *= -1

            pose_2d_fn = fn.replace(
                'Videos', 'Poses_D2_Positions'
            ).replace(
                'mp4', 'cdf'
            )
            gt_pose_2d = cdflib.CDF(datadir + '/' + pose_2d_fn)[0][0].reshape(-1, 32, 2)[frameidx]
            crop_size, crop_bb = get_crop_bb(gt_pose_2d, img.shape)
            img = img[crop_bb[1]:crop_bb[3], crop_bb[0]:crop_bb[2], :]
            heatmap = net(image_transforms(img).unsqueeze(0).to('cuda')).cpu()[0].numpy()

            print(gt_pose_2d.shape)
            gt_pose_2d = gt_pose_2d[[0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]]
            gt_pose_2d[:, 0] -= crop_bb[0]
            gt_pose_2d[:, 1] -= crop_bb[1]
            #
            # plt.imshow(img)
            # plt.scatter(gt_pose_2d[:, 0], gt_pose_2d[:, 1], 5., 'b')
            # plt.scatter(img.shape[0] // 2 + p3d_gt[0, 0, :]*img.shape[0] / 1000, img.shape[0] // 2 - p3d_gt[0, 2, :]*img.shape[0] / 1000., 5., 'r')
            # plt.scatter(img.shape[0] // 2 + p3d_ours[0, 0, :]*img.shape[0] / 1000,
            #             img.shape[0] // 2 - p3d_ours[0, 2, :]*img.shape[0] / 1000., 5., 'g')
            # plt.show()
            #
            # gt_pose_2d[:, 0] /= crop_size / 64.
            # gt_pose_2d[:, 1] /= crop_size / 64.
            #
            # x_center_3d = p3d_gt[0, 0].mean()
            # y_center_3d = -p3d_gt[0, 2].mean()
            #
            # x_center_2d = gt_pose_2d[:, 0].mean()
            # y_center_2d = gt_pose_2d[:, 1].mean()
            #
            # scale_3d = np.sqrt(np.sum((p3d_gt[0, [0, 2]].copy() - p3d_gt[0, [0, 2]].copy().mean(axis=0, keepdims=True))**2))
            # scale_2d = np.sqrt(np.sum((gt_pose_2d - gt_pose_2d.mean(axis=0, keepdims=True))**2))
            #
            # # scale_3d = np.sqrt(scale_3d.mean())
            # # scale_2d = np.sqrt(scale_2d.mean())
            #
            # print(scale_3d, scale_2d)
            # print(x_center_3d, y_center_3d)
            # print(x_center_2d, y_center_2d)
            #
            # x_sharma = (p3d_sharma[:, 0] - x_center_3d) / scale_3d
            # y_sharma = (-p3d_sharma[:, 2] - y_center_3d) / scale_3d
            #
            # x_wehrbein = (p3d_wehrbein[:, 0] - x_center_3d) / scale_3d
            # y_wehrbein = (-p3d_wehrbein[:, 2] - y_center_3d) / scale_3d
            #
            # x_ours = (p3d_ours[:, 0] - x_center_3d) / scale_3d
            # y_ours = (-p3d_ours[:, 2] - y_center_3d) / scale_3d
            #
            # x_gt = (p3d_gt[0, 0] - x_center_3d) / scale_3d
            # y_gt = (-p3d_gt[0, 2] - y_center_3d) / scale_3d
            #
            # x_wehrbein = x_center_2d + x_wehrbein * scale_2d
            # y_wehrbein = y_center_2d - y_wehrbein * scale_2d
            #
            # x_ours = x_center_2d + x_ours * scale_2d
            # y_ours = y_center_2d - y_ours * scale_2d
            #
            # x_gt = x_center_2d + x_gt * scale_2d
            # y_gt = y_center_2d + y_gt * scale_2d
            #
            # plt.imshow(img)
            # plt.scatter(gt_pose_2d[:, 0], gt_pose_2d[:, 1], 5., 'b')
            # plt.scatter(x_gt,
            #             y_gt, 5., 'r')
            # plt.scatter(x_ours.mean(axis=0),
            #             y_ours.mean(axis=0), 5., 'g')
            # plt.show()

            x_wehrbein = p3d_wehrbein[:, 0]
            y_wehrbein = -p3d_wehrbein[:, 1]

            x_ours = p3d_ours[:, 0]
            y_ours = -p3d_ours[:, 1]

            print(p3d_gt.shape)
            x_gt = p3d_gt[0, 0]
            y_gt = -p3d_gt[0, 1]


            print(gt_pose_2d.shape, x_gt.shape)
            print(gt_pose_2d)


            x_max = x_gt.max()
            x_min = x_gt.min()

            x_center = (x_max + x_min) / 2
            x_diff = np.abs(x_max - x_min)

            y_min = y_gt.max()
            y_max = y_gt.min()
            y_center = (y_max + y_min) / 2
            y_diff = np.abs(y_max - y_min)

            scale = max(x_diff, y_diff) * 1.2

            x_ours = (x_ours - x_center) / scale
            y_ours = (y_ours - y_center) / scale

            print(x_center, y_center, scale)
            # x_sharma = (x_sharma - x_center) / scale
            # y_sharma = (y_sharma - y_center) / scale

            overlayed_img = 0.
            cmap = color_map(17, normalized=True)
            for i in range(heatmap.shape[0]):
                color = cmap[i + 1][None, None, :]
                colored_heatmap = color * heatmap[i][:, :, None]
                overlayed_img += colored_heatmap

                plt.imshow(colored_heatmap, vmin=0., vmax=1.)
                # plt.scatter(
                #     x_center_2d + scale_2d * x_sharma[:, H36M17j_TO_MPII[i]],
                #     y_center_2d + scale_2d * y_sharma[:, H36M17j_TO_MPII[i]],
                #     1.,
                #     'r'
                # )
                print(x_wehrbein.shape, x_ours.shape, x_gt.shape)
                plt.scatter(
                    32 + 64 * x_wehrbein[:, H36M17j_TO_MPII[i]],
                    32 + 64 * y_wehrbein[:, H36M17j_TO_MPII[i]],
                    1.,
                    'y'
                )
                plt.scatter(
                    32 + 64 * x_ours[:, H36M17j_TO_MPII[i]],
                    32 + 64 * y_ours[:, H36M17j_TO_MPII[i]],
                    1.,
                    'g'
                )
                plt.scatter(
                    32 + 64 * x_gt[H36M17j_TO_MPII[i]],
                    32 + 64 * y_gt[H36M17j_TO_MPII[i]],
                    10.,
                    'b',
                )
                plt.title('Joint {} ({})'.format(i, H36M_NAMES[H36M17j_TO_MPII[i]]))
                plt.show()
            exit()


            plt.imshow(overlayed_img)
            plt.show()
            exit()


            pose_3d_others = poses_3d_others[:, hardsetidx_alt]
            # plot17j_multi(np.asarray(pose_3d_others.reshape(-1, 3*17)[:10]))



if __name__ == '__main__':
    plot_qualitative_results_based_on_csv(
        'hardest_samples_h36m.csv',
        '/media/karl/Samsung_T5/data/h36m/extracted',
        wehrbein='results/wehrbein_hard_set_results_1000hypos_part1.npy',
        sharma='results/sharma_hard_set_results1000hypos.pickle',
        ours='results/samples_2jyksd3m/hard_samples_poses.npy',
        gt='results/samples_2jnwqazh/hard_samples_gt.npy'
    )




