#import wandb

def plot_real17j_1f_scaled(poses, gt=None, log_wandb=False, title=None):
    import matplotlib as mpl
    #mpl.use('Qt5Agg')
    #mpl.use('GTK3Agg')

    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.animation as anim

    from mpl_toolkits.mplot3d import axes3d, Axes3D

    fig = plt.figure()

    plot_idx = 1

    frames = np.linspace(start=0, stop=poses.shape[0] - 1, num=10).astype(int)

    ax = fig.gca(projection='3d')

    pose = poses[0]

    pose = (pose.reshape(3, 17) - pose.reshape(3, 17)[:, [0]]).reshape(51)
    gt = (gt.reshape(3, 17) - gt.reshape(3, 17)[:, [0]]).reshape(51)

    scale = np.linalg.norm(gt) / np.linalg.norm(pose)
    pose = pose * scale

    bone_map = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12],
                [12, 13], [8, 14], [14, 15], [15, 16]]

    if gt is not None:
        alpha = 0.2
        ax.scatter(gt[0:17], gt[17:34], gt[34:51], c=['#ff0000'], alpha=alpha)
        x = gt[0:17]
        y = gt[17:34]
        z = gt[34:51]
        ax.plot(x[([0, 1])], y[([0, 1])], z[([0, 1])], color='green', alpha=alpha)
        ax.plot(x[([1, 2])], y[([1, 2])], z[([1, 2])], color='green', alpha=alpha)
        ax.plot(x[([2, 3])], y[([2, 3])], z[([2, 3])], color='green', alpha=alpha)
        ax.plot(x[([0, 4])], y[([0, 4])], z[([0, 4])], color='red', alpha=alpha)
        ax.plot(x[([4, 5])], y[([4, 5])], z[([4, 5])], color='red', alpha=alpha)
        ax.plot(x[([5, 6])], y[([5, 6])], z[([5, 6])], color='red', alpha=alpha)
        ax.plot(x[([0, 7])], y[([0, 7])], z[([0, 7])], color='black', alpha=alpha)
        ax.plot(x[([7, 8])], y[([7, 8])], z[([7, 8])], color='black', alpha=alpha)
        ax.plot(x[([8, 9])], y[([8, 9])], z[([8, 9])], color='black', alpha=alpha)
        ax.plot(x[([9, 10])], y[([9, 10])], z[([9, 10])], color='black', alpha=alpha)
        ax.plot(x[([8, 11])], y[([8, 11])], z[([8, 11])], color='green', alpha=alpha)
        ax.plot(x[([11, 12])], y[([11, 12])], z[([11, 12])], color='green', alpha=alpha)
        ax.plot(x[([12, 13])], y[([12, 13])], z[([12, 13])], color='green', alpha=alpha)
        ax.plot(x[([8, 14])], y[([8, 14])], z[([8, 14])], color='red', alpha=alpha)
        ax.plot(x[([14, 15])], y[([14, 15])], z[([14, 15])], color='red', alpha=alpha)
        ax.plot(x[([15, 16])], y[([15, 16])], z[([15, 16])], color='red', alpha=alpha)

    x = pose[0:17]
    y = pose[17:34]
    z = pose[34:51]
    ax.scatter(x, y, z)

    alpha = 1.0
    ax.plot(x[([0, 1])], y[([0, 1])], z[([0, 1])], color='green', alpha=alpha)
    ax.plot(x[([1, 2])], y[([1, 2])], z[([1, 2])], color='green', alpha=alpha)
    ax.plot(x[([2, 3])], y[([2, 3])], z[([2, 3])], color='green', alpha=alpha)
    ax.plot(x[([0, 4])], y[([0, 4])], z[([0, 4])], color='red', alpha=alpha)
    ax.plot(x[([4, 5])], y[([4, 5])], z[([4, 5])], color='red', alpha=alpha)
    ax.plot(x[([5, 6])], y[([5, 6])], z[([5, 6])], color='red', alpha=alpha)
    ax.plot(x[([0, 7])], y[([0, 7])], z[([0, 7])], color='black', alpha=alpha)
    ax.plot(x[([7, 8])], y[([7, 8])], z[([7, 8])], color='black', alpha=alpha)
    ax.plot(x[([8, 9])], y[([8, 9])], z[([8, 9])], color='black', alpha=alpha)
    ax.plot(x[([9, 10])], y[([9, 10])], z[([9, 10])], color='black', alpha=alpha)
    ax.plot(x[([8, 11])], y[([8, 11])], z[([8, 11])], color='green', alpha=alpha)
    ax.plot(x[([11, 12])], y[([11, 12])], z[([11, 12])], color='green', alpha=alpha)
    ax.plot(x[([12, 13])], y[([12, 13])], z[([12, 13])], color='green', alpha=alpha)
    ax.plot(x[([8, 14])], y[([8, 14])], z[([8, 14])], color='red', alpha=alpha)
    ax.plot(x[([14, 15])], y[([14, 15])], z[([14, 15])], color='red', alpha=alpha)
    ax.plot(x[([15, 16])], y[([15, 16])], z[([15, 16])], color='red', alpha=alpha)

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())

    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    # ax.axis('equal')
    #ax.axis('off')
    plt.xlabel('x')
    plt.ylabel('y')

    if title is not None:
        plt.title(title)

    if log_wandb:
        wandb.log({"chart": plt})
    else:
        plt.show()

    return plt
