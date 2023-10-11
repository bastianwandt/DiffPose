# Data Description
The following is a short overview of the structure of the generated data and description of the content.

- Each subject has a separate folder containing each of the sequences which it's part of
- Each sequence folder contain a number of .npy files (numpy arrays saved through np.save) + a json file.
- The json file contains:

    - a "Meta" entry with action name and the filename of the sequence it corresponds to.
    - A "Frames" entry containing a list of "Frames disctionaries" with the following entries
        - "FrameIndex": Which frame in the sequence it corresponds to (Matches the .npy file with the same FrameIndex.
        - "CropBB": The BB box used for cropping the original image (Obs, upper coord is not included in the ROI)
        - "Joints 2D GT": List of the ground-truth 2D joints position from "Poses_D2_Positions" (OBS, 32 joints are defined rather than the 16 used in MPII or the 17 used in H36M)
            - The 2D joints are transformed to the ROI and normalized according to:
                - joints_2d -= upper_left_corner_of_ROI
                - joints_2d /= crop_size
        - "Joints 3d GT MONO": List of the ground-truth 3D joints position from "Poses_D3_Positions_mono" (OBS. 32 joints)
        - The joints are stored through np.tolist() so to read them as a numpy array, use np.asarray(joint_list)

- The Heatmaps are the raw heatmaps from the hrnet of shape (16, 64, 64)


# Data generation
The following describes what should be done in order to be able to generate new heatmaps

- Make sure that the H36M dataset has been downloaded and extracted (I.e. using download_all, extract_all from https://github.com/anibali/h36m-fetch)
- Modify the "DATA_DIR" and the "OUTPUT_DIR" variables in "hrnet/mpii_hrnet_w32_255x255.yaml" accordingly
- (Optinal) Modify skip_frames in pose_detection.py if the number of frames to skip shouldn't be 16
- Make sure all dependencies are installed and run H36M.py from this directory (for the Human 3.6M dataset, other datasets are named similarly)

### Troubleshooting guide
- Make sure that "git lfs" is available and installed, otherwise the network weights will be an empty file. 
  - Best way to avoid this is to check the reported file size of fine_HRNet.pt (Should be about 110MB)

### Future improvements
List some possible improvements on the TODO list

- Include the argmax output from the heatmaps in the json file
- Change the precision of the stored heatmaps to float16 instead of float32 to save space


# Short description of how things are calculated and extracted

# Image preprocessing:
Following the preprocessing used in "Probabilistic Monocular 3D Human Pose Estimation with Normalizing Flows" 
by Tom Whrbein et. al. the following is done:
### Cropping:
A rectangular Region-of-interest in each image is extracted by calculating the min and max x/y position for each joint 
and expanding the width and height by a factor 1.2, e.g.

    # Code for a single pose with joints of shape (2x16)
    bb_min = min(joints, axis=1)
    bb_max = max(joints, axis=1)
    bb_width = bb_max[0] - bb_min[0]
    bb_height = bb_max[1] - bb_min[1]
    
    bb_center = round((bb_max+bb_min) / 2)

    cropsize = round(max(bb_width, bb_height) * 1.2)
    crop_bb = round([
        bb_center[0] - cropsize*0.5,
        bb_center[1] - cropsize*0.5,
        bb_center[0] + cropsize*0.5,
        bb_center[1] + cropsize*0.5
    ])

### Pose detection and heat map generation:
The pose is estimated using a pretrained HRnet, fine-tuned on the Human 3.6M dataset.

The code is available here:
https://github.com/HRNet/HRNet-Human-Pose-Estimation/blob/master/README.md

And this is the fine-tuned model weights:
https://drive.google.com/file/d/1AgeflLDRudx3qRqvMkjyrkWyeDFHwWu6/view?usp=sharing
