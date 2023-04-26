import PIL.Image
import torch
from torchvision import transforms
import tables as tb
from pathlib import Path
import cdflib
import imageio
import numpy as np
from tqdm import auto

from utils import get_pretrained_model, get_crop_bb


# correct keys:
subjects_train = ['S1', 'S5', 'S6', 'S7', 'S8']
subjects_test = ['S9', 'S11']
actions_all = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases',
               'Sitting', 'SittingDown', 'Smoking', 'Waiting', 'WalkDog', 'WalkTogether', 'Walking']
subactions_all = ['0', '1']
cameras_all = ['54138969', '55011271', '58860488', '60457274']


def fix_dot_2(infos):
    if '2.' in infos:
        infos = infos.replace("2.", "1.")
    return infos


def fix_dot_3(infos):
    if '3.' in infos:
        infos = infos.replace("3.", "1.")
    return infos


def fix_1_2(infos):
    if '2.' in infos:
        infos = infos.replace("2.", "")
    return infos


def fix_2_3(infos):
    if '2.' in infos:
        infos = infos.replace("2.", "1.")
    if '3.' in infos:
        infos = infos.replace("3.", "")
    return infos


class DatasetEntry(tb.IsDescription):
    action = tb.StringCol(16)
    subject = tb.StringCol(3)
    subaction = tb.StringCol(1)
    cam = tb.StringCol(8)

    frame_orig = tb.IntCol(4)
    crop_size = tb.IntCol(4)

    hardsubset = tb.BoolCol(1)

    gt_3d = tb.Float32Col(shape=(3, 17))
    gt_2d = tb.Float32Col(shape=(2, 17))
    argmax_2d = tb.Float32Col(shape=(2, 16))
    gt_2d_image_coords = tb.Float32Col(shape=(2, 32))

    heatmap = tb.Float16Col(shape=(16, 64, 64))


def main(args):
    model, cfg = get_pretrained_model(use_h36m_model=not args.use_orig_hrnet)
    model.to('cuda')
    model.eval()

    if args.use_orig_hrnet:
        hrnet_suffix = 'orig_hrnet'
    else:
        hrnet_suffix = 'h36m_hrnet'
    datadir = Path(args.input_dir)
    outputdir = Path(args.output_dir)

    with open("framelist_h36m_hard_subset", "r") as fn_subset:
        hard_subset_list = fn_subset.readlines()
    hard_subset_list = [x.strip('\n') for x in hard_subset_list]

    outputdir.mkdir(exist_ok=True, parents=True)
    h5file = tb.open_file(outputdir.joinpath("H36M_dataset_{}.h5".format(hrnet_suffix)), mode="w", title="Dataset H36M and H36MA with heatmaps")

    group = h5file.create_group('/', 'dataset', 'Dataset')
    table_train = h5file.create_table(group, 'trainset', DatasetEntry, "Training set")
    table_test = h5file.create_table(group, 'testset', DatasetEntry, "Test set")

    entry_train = table_train.row

    generate_heatmaps_and_fill_dataset(entry_train, model, "train", datadir, hard_subset_list)
    table_train.flush()
    entry_test = table_test.row
    generate_heatmaps_and_fill_dataset(entry_test, model, "test", datadir, hard_subset_list)


def generate_heatmaps_and_fill_dataset(entry, model, split, datadir, hard_subset_list):
    # Define the image_transforms used in the original code (+resize)
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
        transforms.Resize([255, 255])
    ])

    if split == 'train':
        subject_list = subjects_train
    else:
        subject_list = subjects_test

    for subject in subject_list:
        subject_dir = datadir.joinpath(subject)
        pose_2d_dir = subject_dir.joinpath("Poses_D2_Positions")
        pose_3d_dir = subject_dir.joinpath("Poses_D3_Positions_mono")
        pose_3d_univ_dir = subject_dir.joinpath("Poses_D3_Positions_mono_universal")
        video_dir = subject_dir.joinpath("Videos")

        hard_subset_list_for_subject = [x for x in hard_subset_list if subject == x.split('/')[1]]

        for sequence in auto.tqdm(sorted(pose_2d_dir.iterdir()), desc='Sequence'):
            # Number of files are not identical so we iter one directory and find the corresponding file from the names
            filename = sequence.stem
            if ' ' in filename:
                action, rest_info = filename.split(" ")
            else:
                action, rest_info = filename.split(".")

            action, rest_info, cam, subact = rename_sequences_for_consistencies(action, rest_info, subject)

            if subject == 'S5' and subact == '1' and action == 'Waiting' and cam == '55011271':
                continue
            if subject == 'S11' and subact == '0' and action == 'Directions' and cam == '54138969':
                continue

            hard_subset_list_for_sequence = [
                x for x in hard_subset_list_for_subject if sequence.stem == x.split('/')[2]
            ]

            #used_frames = len(dataset[subject][action][subact][cam]['imgpath'])
            poses_2d = cdflib.CDF(sequence)[0][0]
            poses_3d = cdflib.CDF(pose_3d_dir.joinpath(sequence.stem + '.cdf'))[0][0]
            poses_3d_uni = cdflib.CDF(pose_3d_univ_dir.joinpath(sequence.stem + '.cdf'))[0][0]

            assert poses_3d.shape[1] == 96
            assert poses_3d_uni.shape[1] == 96
            assert poses_2d.shape[1] == 64
            # select 17 joints:
            joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
            poses_3d = poses_3d.reshape(-1, 32, 3)[:, joints]
            poses_3d = poses_3d.swapaxes(1, 2) #.reshape(-1, 3 * 17)

            poses_3d_uni = poses_3d_uni.reshape(-1, 32, 3)[:, joints]
            poses_3d_uni = poses_3d_uni.swapaxes(1, 2) #.reshape(-1, 3 * 17)

            poses_2d_gt = poses_2d.reshape(-1, 32, 2)[:, joints]
            poses_2d_gt = poses_2d_gt.swapaxes(1, 2) #.reshape(-1, 2 * 17)

            video_file = video_dir.joinpath(sequence.stem + '.mp4')
            # Assuming that each pose match the same frame with the same index
            cap = imageio.get_reader(video_file, 'ffmpeg')

            n_frames = 0
            for idx, (img, pose) in enumerate(zip(cap, poses_2d)):
                # Start from frame 3 as in https://github.com/twehrbein/Probabilistic-Monocular-3D-Human-Pose-Estimation-with-Normalizing-Flows/blob/main/create_data.py
                if (idx + 1) % 4 != 0:
                    continue

                img = PIL.Image.fromarray(img)
                joints = pose.reshape(32, 2)

                crop_size, crop_bb = get_crop_bb(joints, img.size[::-1])

                # Crop the ROI from the image
                roi = img.crop(crop_bb)
                # roi.show()
                roi = np.asarray(roi).copy()

                # Update the joint positions, given the cropping region
                joints[:, 0] -= crop_bb[0]
                joints[:, 1] -= crop_bb[1]

                # Rescale the joint positiongs to the interval [0, 1]
                joints /= crop_size

                # Generate joint position predictions (OBS! only 16 joints are predicted which differs from the given 32 joints in the dataset)
                # Joints defined by MPII
                preprocessed_img = image_transforms(roi)
                with torch.inference_mode():
                    prediction = model(preprocessed_img.unsqueeze(0).cuda())[0].cpu()

                # Convert predictions to numpy and change to float16 precision
                prediction = prediction.numpy().astype(np.float16)

                # Naively find the argmax of each joint in 64x64 map and normalize to [0, 1] coords
                peak = np.unravel_index(np.argmax(prediction.reshape(prediction.shape[0], -1), axis=1), prediction.shape[1:])
                peak_np = np.asarray(peak, dtype=np.float) / prediction.shape[1]

                # Fill the data entry with all necessary data
                entry["heatmap"] = prediction
                entry["subject"] = subject
                entry["action"] = action
                entry["subaction"] = subact
                entry["cam"] = cam
                entry["frame_orig"] = idx
                entry["crop_size"] = crop_size

                entry["gt_3d"] = poses_3d[idx, :]
                entry["gt_2d"] = poses_2d_gt[idx, :]
                entry["gt_2d_image_coords"] = joints.swapaxes(0, 1)
                entry['argmax_2d'] = peak_np

                # Add one since the subset_list index the frames starting from 1 instead of 0
                if "/{}/{}/{}.jpg".format(subject, sequence.stem, idx+1) in hard_subset_list_for_sequence:
                    entry["hardsubset"] = True
                else:
                    entry["hardsubset"] = False

                entry.append()
                n_frames += 1
    return entry


def rename_sequences_for_consistencies(action, rest_info, subject):
    # rename for consistency:
    # TakingPhoto -> Photo, WalkingDog -> WalkDog
    if action == 'TakingPhoto':
        action = 'Photo'
    if action == 'WalkingDog':
        action = 'WalkDog'

    # take care of inconsistent naming...
    if subject == 'S1':
        if action == 'Eating':
            # S1 Eating (., 2)
            rest_info = fix_dot_2(rest_info)
        if action == 'Sitting':
            # S1 Sitting (1, 2)
            rest_info = fix_1_2(rest_info)
        if action == 'SittingDown':
            # S1 SittingDown (., 2)
            rest_info = fix_dot_2(rest_info)

    if subject == 'S5':
        if action == 'Directions':
            # S5 Directions (1, 2)
            rest_info = fix_1_2(rest_info)
        if action == 'Discussion':
            # S5 Discussion (2, 3)
            rest_info = fix_2_3(rest_info)
        if action == 'Greeting':
            # S5 Greeting (1, 2)
            rest_info = fix_1_2(rest_info)
        if action == 'Photo':
            # S5 Photo (., 2)
            rest_info = fix_dot_2(rest_info)
        if action == 'Waiting':
            # S5 Waiting (1, 2)
            rest_info = fix_1_2(rest_info)

    if subject == 'S6':
        if action == 'Eating':
            # S6 Eating (1, 2)
            rest_info = fix_1_2(rest_info)
        if action == 'Posing':
            # S6 Posing (., 2)
            rest_info = fix_dot_2(rest_info)
        if action == 'Sitting':
            # S6 Sitting (1,2)
            rest_info = fix_1_2(rest_info)
        if action == 'Waiting':
            # S6 Waiting (., 3)
            rest_info = fix_dot_3(rest_info)

    if subject == 'S7':
        if action == 'Phoning':
            # S7 Phoning (., 2)
            rest_info = fix_dot_2(rest_info)
        if action == 'Waiting':
            # S7 Waiting (1, 2)
            rest_info = fix_1_2(rest_info)
        if action == 'Walking':
            # S7 Walking (1, 2)
            rest_info = fix_1_2(rest_info)

    if subject == 'S8':
        if action == 'WalkTogether':
            # S8 WalkTogether (1, 2)
            rest_info = fix_1_2(rest_info)

    if subject == 'S9':
        if action == 'Discussion':
            # S9 discussion (1, 2)
            rest_info = fix_1_2(rest_info)

    if subject == 'S11':
        if action == 'Discussion':
            rest_info = fix_1_2(rest_info)
        if action == 'Greeting':
            # S11 greeting (., 2)
            rest_info = fix_dot_2(rest_info)
        if action == 'Phoning':
            # S11 phoning (2,3)
            rest_info = fix_2_3(rest_info)
        if action == 'Smoking':
            # S11 smoking (., 2)
            if '2.' in rest_info:
                # replace 2. with .
                rest_info = fix_dot_2(rest_info)

    assert rest_info[:2] == '1.' or '.' not in rest_info
    if '.' not in rest_info:
        subact = '0'
        cam = rest_info
    else:
        subact = '1'
        cam = rest_info.split('.')[-1]

    return action, rest_info, cam, subact


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Input directory in which the subject data resides')
    parser.add_argument('--output_dir', type=str, help='Output directory to which the dataset will stored')

    parser.add_argument('--use_orig_hrnet', action='store_true', help='Use original hrnet weights instead of the weights pretrained on Human3.6M')

    _args = parser.parse_args()

    main(_args)
