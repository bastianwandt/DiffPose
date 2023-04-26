import torch
from torchvision import transforms
import tables as tb
import pickle
from PIL import Image
from pathlib import Path
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


class DatasetEntry(tb.IsDescription):
    action = tb.StringCol(16)
    subject = tb.StringCol(2)
    subaction = tb.StringCol(1)
    cam = tb.StringCol(8)

    frame_orig = tb.IntCol(4)

    hardsubset = tb.BoolCol(1)

    gt_3d = tb.Float32Col(shape=(24, 3))
    gt_2d = tb.Float32Col(shape=(3, 18))
    argmax_2d = tb.Float32Col(shape=(2, 16))

    heatmap = tb.Float16Col(shape=(16, 64, 64))


def main(args):
    """Get pretrained 2D detector model"""
    model, cfg = get_pretrained_model(use_h36m_model=not args.use_orig_hrnet)
    model.to('cuda')
    model.eval()

    if args.use_orig_hrnet:
        hrnet_suffix = 'orig_hrnet'
    else:
        hrnet_suffix = 'h36m_hrnet'
    datadir = Path(args.input_dir)
    outputdir = Path(args.output_dir)

    outputdir.mkdir(exist_ok=True, parents=True)
    h5file = tb.open_file(outputdir.joinpath("3DPW_dataset_{}.h5".format(hrnet_suffix)), mode="w", title="Dataset 3DPW with heatmaps")

    group = h5file.create_group('/', 'dataset', 'Dataset')
    table_train = h5file.create_table(group, 'trainset', DatasetEntry, "Training set")
    table_test = h5file.create_table(group, 'testset', DatasetEntry, "Test set")

    entry_train = table_train.row

    generate_heatmaps_and_fill_dataset(entry_train, model, "train", datadir)
    table_train.flush()
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
        subjects = ['train']#, 'validation']
    else:
        subjects = ['test']

    for subject in subjects:
        sequence_dir = datadir.joinpath('sequenceFiles', subject)
        image_dir = datadir.joinpath("imageFiles")

        for sequence in auto.tqdm(sorted(sequence_dir.iterdir()), desc='Sequence'):
            print('Processing seq:{} ({})'.format(sequence, subject))
            u = pickle._Unpickler(open(sequence, 'rb'))
            u.encoding = 'latin1'
            data = u.load()

            seq_name = data['sequence']
            img_frames = data['img_frame_ids']  # Not in use
            for person_idx in range(len(data['poses2d'])):
                poses2d = data['poses2d'][person_idx]  # Shape Nx3x18
                poses3d = data['jointPositions'][person_idx].reshape(-1, 24, 3)
                img_dir = datadir.joinpath("imageFiles", seq_name)

                for fnr, (p2d, p3d) in enumerate(zip(poses2d, poses3d)):
                    img_fn = img_dir.joinpath('image_{:05d}.jpg'.format(fnr))
                    img = Image.open(img_fn)

                    if np.all(p2d[2] == 0):
                        continue

                    crop_size, crop_bb = get_crop_bb(p2d.transpose(), img.size[::-1])
                    roi = img.crop(crop_bb)
                    roi = np.asarray(roi).copy()

                    preprocessed_img = image_transforms(roi)
                    with torch.inference_mode():
                        prediction = model(preprocessed_img.unsqueeze(0).cuda())[0].cpu()

                    # Convert predictions to numpy and change to float16 precision
                    prediction = prediction.numpy().astype(np.float16)

                    # Naively find the argmax of each joint in 64x64 map and normalize to [0, 1] coords
                    peak = np.unravel_index(np.argmax(prediction.reshape(prediction.shape[0], -1), axis=1),
                                            prediction.shape[1:]) / prediction.shape[1]

                    # Fill the data entry with all necessary data
                    entry["heatmap"] = prediction
                    entry["subject"] = subject
                    entry["action"] = "none"
                    entry["subaction"] = "none"
                    entry["cam"] = "none"
                    entry["frame_orig"] = fnr

                    entry["gt_3d"] = p3d
                    entry["gt_2d"] = p2d
                    entry['argmax_2d'] = peak

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