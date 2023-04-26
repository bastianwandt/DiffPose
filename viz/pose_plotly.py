import numpy as np
import plotly.express as px


def get_multiple_plotly_pose_figs(poses, pose_gt=None):
    # pickle.dump(poses, open("res/samples3d_np.pkl", "wb"))

    body_parts = {
        "left_leg": np.array([0, 1, 2, 3]),
        "right_leg": np.array([0, 4, 5, 6]),
        "spine": np.array([0, 7, 8, 9, 10]),
        "right_arm": np.array([8, 11, 12, 13]),
        "left_arm": np.array([8, 14, 15, 16]),
    }
    print(poses.shape)
    if pose_gt is not None:
        print(pose_gt.shape)
    joints = poses.reshape(poses.shape[0], 3, 17)

    connected_joints = []
    colors = []

    for p in range(poses.shape[0]):
        connected_joints.append(
            np.concatenate([joints[p, :, body_parts[part]] for part in body_parts], axis=0)
        )
        colors.append(
            np.concatenate([[part + " " + str(p)] * len(body_parts[part]) for i, part in enumerate(body_parts)], axis=0)
        )

    if pose_gt is not None:
        joints_gt = pose_gt.reshape(3, 17)
        connected_joints.append(
            np.concatenate([joints_gt[:, body_parts[part]].transpose() for part in body_parts], axis=0)
        )
        colors.append(
            np.concatenate([[part + " GT"] * len(body_parts[part]) for i, part in enumerate(body_parts)], axis=0)
        )

    connected_joints = np.concatenate(connected_joints, axis=0)
    colors = np.concatenate(colors, axis=0)

    if pose_gt is not None:
        title = 'Pose w GT'
    else:
        title = 'Sampled Poses'
    fig = px.line_3d(
        x=connected_joints[:, 0],
        y=connected_joints[:, 1],
        z=connected_joints[:, 2],
        color=colors,
        markers=True,
        title=title
    )
    fig.update_traces(marker=dict(
        size=2
    ))

    return fig


def get_plotly_pose_fig(pose):
    # pickle.dump(poses, open("res/samples3d_np.pkl", "wb"))

    body_parts = {
        "left_leg": np.array([0, 1, 2, 3]),
        "right_leg": np.array([0, 4, 5, 6]),
        "spine": np.array([0, 7, 8, 9, 10]),
        "right_arm": np.array([8, 11, 12, 13]),
        "left_arm": np.array([8, 14, 15, 16]),
    }

    joints = pose.reshape(3, 17)

    connected_joints = np.concatenate([joints[:, body_parts[part]] for part in body_parts], axis=1)
    colors = np.concatenate([[part] * len(body_parts[part]) for i, part in enumerate(body_parts)], axis=0)

    fig = px.line_3d(
        x=connected_joints[0, :],
        y=connected_joints[1, :],
        z=connected_joints[2, :],
        color=colors,
        markers=True
    )
    fig.update_traces(marker=dict(
        size=2
    ))

    return fig


if __name__ == '__main__':
    import pickle

    use_wandb = False
    sampled_poses = pickle.load(open("../res/samples3d.pkl", "rb"))
    poses = sampled_poses.cpu().numpy()

    if use_wandb:
        import wandb

        wandb.init(
            project="testing plotly",
            name="plots",
            entity="probabilistic_human_pose"
        )

    # Generate a plot for a single pose
    fig = get_plotly_pose_fig(poses[0])

    print(fig.data, fig.layout)

    if use_wandb:
        wandb.log({"pose": fig})
    else:
        fig.show()

    # Generate a plot for multiple poses
    fig = get_multiple_plotly_pose_figs(poses[0:20])


    if use_wandb:
        wandb.log({"poses": fig})
        wandb.finish()
    else:
        fig.show()

