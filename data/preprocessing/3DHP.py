import torch
from torchvision import transforms
import tables as tb
from pathlib import Path
import numpy as np

from utils import get_pretrained_model, get_crop_bb


# correct keys:
subjects_train = []
subjects_test = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']


class DatasetEntry(tb.IsDescription):
    action = tb.StringCol(4)
    subject = tb.StringCol(4)
    subaction = tb.StringCol(1)
    cam = tb.StringCol(8)

    frame_orig = tb.IntCol(4)

    hardsubset = tb.BoolCol(1)

    gt_3d = tb.Float32Col(shape=(3 * 17))
    gt_3d_uni = tb.Float32Col(shape=(3 * 17))
    gt_2d = tb.Float32Col(shape=(2 * 17))
    argmax_2d = tb.Float32Col(shape=(2 * 16))

    heatmap = tb.Float16Col(shape=(16, 64, 64))


def main(args):
    """Get pretrained 2D detector model"""
    model, cfg = get_pretrained_model(use_h36m_model=not args.use_orig_hrnet)
    model.to('cuda')
    model.eval()

    datadir = Path(args.input_dir)
    outputdir = Path(args.output_dir)

    if args.use_orig_hrnet:
        hrnet_suffix = 'orig_hrnet'
    else:
        hrnet_suffix = 'h36m_hrnet'
    outputdir.mkdir(exist_ok=True, parents=True)
    h5file = tb.open_file(outputdir.joinpath("3DHP_dataset_{}.h5".format(hrnet_suffix)), mode="w", title="Dataset 3DHP with heatmaps")

    group = h5file.create_group('/', 'dataset', 'Dataset')
    table_test = h5file.create_table(group, 'testset', DatasetEntry, "Test set")

    entry_test = table_test.row
    generate_heatmaps_and_fill_dataset(entry_test, model, "test", datadir)


def generate_heatmaps_and_fill_dataset(entry, model, split, datadir, sample_interesting_frames=False):
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

        data_annotations = tb.open_file(subject_dir.joinpath('annot_data.mat'), 'r')
        video_dir = subject_dir.joinpath('imageSequence')

        images = sorted(video_dir.iterdir())

        print('Found {} images'.format(len(images)))

        bb_crop = data_annotations.root.bb_crop
        poses_3d_uni = data_annotations.root.univ_annot3  # There's also annot3
        poses_3d = data_annotations.root.annot3  # There's also annot3
        poses_2d = data_annotations.root.annot2  # There's also annot3
        valid_frames = data_annotations.root.valid_frame

        print('num valid frames:', np.sum(valid_frames))

        for idx, (fn_img, pose_3d_uni, pose_3d, pose_2d, valid) in enumerate(zip(images, poses_3d_uni, poses_3d, poses_2d, valid_frames)):
            if not valid:
                continue

            from PIL import Image
            img = Image.open(fn_img, 'r')
            joints = pose_2d.reshape(17, 2)
            crop_size, crop_bb = get_crop_bb(joints, img.size[::-1])

            """Will zero pad image if close to image border"""
            roi = img.crop(crop_bb)
            roi = np.asarray(roi).copy()

            """Update the joint positions, given the cropping region
            Translates and scales to image normalized coordinates [0, 1]
            """
            joints[:, 0] -= crop_bb[0]
            joints[:, 1] -= crop_bb[1]
            joints[:, 0] /= crop_size
            joints[:, 1] /= crop_size

            preprocessed_img = image_transforms(roi)
            with torch.inference_mode():
                prediction = model(preprocessed_img.unsqueeze(0).cuda())[0].cpu()

            # Convert predictions to numpy and change to float16 precision
            prediction = prediction.numpy()

            # Naively find the argmax of each joint in 64x64 map and normalize to [0, 1] coords
            peak = np.unravel_index(np.argmax(prediction.reshape(prediction.shape[0], -1), axis=1),
                                    prediction.shape[1:]) / crop_size

            # Fill the data entry with all necessary data
            entry["heatmap"] = prediction.astype(np.float16)
            entry["subject"] = subject
            entry["action"] = subject  # Missuse of action in order to reuse evaluation code
            entry["frame_orig"] = idx

            ordering = [14, 8, 9, 10, 11, 12, 13, 15, 1, 16, 0, 5, 6, 7, 2, 3, 4]
            pose_2d = np.take(pose_2d, ordering, axis=1)
            pose_3d = np.take(pose_3d, ordering, axis=1)
            pose_3d_uni = np.take(pose_3d_uni, ordering, axis=1)

            entry["gt_3d"] = pose_3d.reshape(17, 3).swapaxes(0, 1).reshape(-1)
            entry["gt_3d_uni"] = pose_3d_uni.reshape(17, 3).swapaxes(0, 1).reshape(-1)
            entry["gt_2d"] = pose_2d.reshape(17, 2).swapaxes(0, 1).reshape(-1)
            entry['argmax_2d'] = peak.reshape(-1)

            entry.append()
    return entry


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Input directory in which the subject data resides')
    parser.add_argument('--output_dir', type=str, help='Output directory to which the dataset will stored')

    parser.add_argument('--use_orig_hrnet', action='store_true',
                        help='Use original hrnet weights instead of the weights pretrained on Human3.6M')

    _args = parser.parse_args()

    main(_args)

