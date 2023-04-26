# Code from: https://github.com/twehrbein/Probabilistic-Monocular-3D-Human-Pose-Estimation-with-Normalizing-Flows/blob/ad2fdf21da1dfb689a5c2a531ebda6e23d95ccc0/utils/eval_functions.py#L66
import torch


def calculate_symmetry_error(poses, reduction='none', dim=-1, njoints=16):
    bones = {
        'h36m': torch.tensor([[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [7, 10], [10, 11],
                           [11, 12], [7, 13], [13, 14], [14, 15]], device=poses.device),
        '3dpw': torch.tensor([[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9],
                      [7, 10], [8, 11], [9, 12], [9, 13], [9, 14], [12, 15], [13, 16], [14, 17],
                      [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23]], device=poses.device)
             }

    # TODO: Fix support for SIMPL joints

    bone_pairs = {'h36m': torch.tensor([[0, 2], [1, 3], [7, 10], [8, 11], [9, 12]], device=poses.device)}
    bone_indices = bones["h36m"] + 1

    start_pos = torch.index_select(poses, -1, bone_indices[:, 0])
    end_pos = torch.index_select(poses, -1, bone_indices[:, 1])

    # Extract bones as delta positions and calculate length
    bone_lengths = torch.linalg.norm(end_pos - start_pos, dim=-2)
    # Find matching bones in skeleton
    bone0 = torch.index_select(bone_lengths, -1, bone_pairs['h36m'][:, 0])
    bone1 = torch.index_select(bone_lengths, -1, bone_pairs['h36m'][:, 1])

    # Calculate the absolute length difference between symmetries
    absolute_error = torch.abs(bone0 - bone1)

    # Calculate the average error for all bones
    absolute_error = absolute_error.mean(dim=-1)

    if reduction == 'none':
        return absolute_error
    elif reduction == 'mean':
        return absolute_error.mean(dim=dim)
    elif reduction == 'sum':
        return absolute_error.sum(dim=dim)


def calculate_ece(p_gt: torch.Tensor, p_pred: torch.Tensor):
    # Calculate Expected Calibration Error (ECE) https://openreview.net/pdf?id=N3FlFslv_J
    # p_gt need to be of shape (-1, 3, #joints)
    # p_pred need to be of shape (-1, n_hypo, 3, #joints)

    # Calculate the median pose per dimension to get a univariate error
    median_pose = torch.quantile(p_pred, q=0.5, dim=1, keepdim=True)

    # Calculate the distance to the mean (skip sorting since we simply count number of elements with a larger error than gt)
    l2_diff = torch.linalg.norm(p_pred - median_pose, dim=-2)

    # Calculate the distance between gt and median pose
    l2_diff_gt = torch.linalg.norm(p_gt - median_pose[:, 0], dim=-2)

    # Get the empirical quantile by calculating the number of elements above and below the gt distance
    quantile = (l2_diff_gt.unsqueeze(1) < l2_diff).float().mean(dim=1)

    # To follow the paper, the quantiles should be averaged across the GT poses followed by the median along the keypoint dimension
    return quantile  # Out size: [bs x 17]


def procrustes_torch_parallel(p_gt, p_pred):
    # p_gt and p_pred need to be of shape (-1, 3, #joints)
    # care: run on cpu! way faster than on gpu

    mu_gt = p_gt.mean(dim=2)
    mu_pred = p_pred.mean(dim=2)

    X0 = p_gt - mu_gt[:, :, None]
    Y0 = p_pred - mu_pred[:, :, None]

    ssX = (X0**2.).sum(dim=(1, 2))
    ssY = (Y0**2.).sum(dim=(1, 2))

    # centred Frobenius norm
    normX = torch.sqrt(ssX) + 1e-6
    normY = torch.sqrt(ssY) + 1e-6

    # scale to equal (unit) norm
    X0 /= normX[:, None, None]
    Y0 /= normY[:, None, None]

    # optimum rotation matrix of Y
    A = torch.bmm(X0, Y0.transpose(1, 2))

    try:
        U, s, V = torch.svd(A, some=True, compute_uv=True)
    except:
        print("ERROR IN SVD, could not converge")
        print("SVD INPUT IS:")
        print(A)
        print(A.shape)
        print('A NaNs:', A.isnan().sum())
        print('GT NaNs', X0.isnan().sum())
        print('Pred NaNs', Y0.isnan().sum())
        print('normX', normX.min(), normX.max())
        print('normY', normY.min(), normY.max())
        return None

    T = torch.bmm(V, U.transpose(1, 2))

    # Make sure we have a rotation
    detT = torch.det(T)
    sign = torch.sign(detT)
    V[:, :, -1] *= sign[:, None]
    s[:, -1] *= sign
    T = torch.bmm(V, U.transpose(1, 2))

    traceTA = s.sum(dim=1)

    # optimum scaling of Y
    b = traceTA * normX / normY

    # standardised distance between X and b*Y*T + c
    d = 1 - traceTA**2

    # transformed coords
    scale = normX*traceTA
    Z = (scale[:, None, None] * torch.bmm(Y0.transpose(1, 2), T) + mu_gt[:, None, :]).transpose(1, 2)

    # transformation matrix
    c = mu_gt - b[:, None]*(torch.bmm(mu_pred[:, None, :], T)).squeeze()

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}
    return d, Z, tform


def err_3dpe_parallel(p_ref, p, return_sum=True, return_poses=False, njoints=16):
    p_ref, p = p_ref.reshape((-1, 3, njoints+1)), p.reshape((-1, 3, njoints+1))
    res = procrustes_torch_parallel(p_ref.clone(), p)
    if res is not None:
        d, Z, tform = res
    else:
        err = torch.ones_like(p_ref.sum(dim=1).sum(dim=1)) * torch.nan
        if return_sum:
            err = err.sum().item()
        if not return_poses:
            return err
        else:
            return err, None
    if return_sum:
        err = torch.sum(torch.mean(torch.sqrt(torch.sum((p_ref - Z)**2, dim=1)), dim=1)).item()
    else:
        err = torch.mean(torch.sqrt(torch.sum((p_ref - Z)**2, dim=1)), dim=1)
    if not return_poses:
        return err
    else:
        return err, Z


def pa_hypo_batch(p_ref, p, njoints=16):
    p_ref, p = p_ref.reshape((-1, 3, njoints+1)), p.reshape((-1, 3, njoints+1))
    d, Z, tform = procrustes_torch_parallel(p_ref.clone(), p.clone())
    return Z


def get_scale(p):
    p_copy = p.clone()
    # mu = p.mean(dim=(-1, -2), keepdim=True)
    # p_copy -= mu
    p_copy -= p_copy[:, :, :, 0, None]


    scale = (p_copy ** 2).sum(dim=(-1, -2), keepdim=True).sqrt()
    return scale


def apply_scale(p, gt_scale=1.):
    p_copy = p.clone()
    # mu = p.mean(dim=(-1, -2), keepdim=True)
    # p_copy -= mu
    p_copy -= p_copy[:, :, :, 0, None]

    scale = (p_copy ** 2).sum(dim=(-1, -2), keepdim=True).sqrt()
    p_copy = p_copy * gt_scale / scale
    p_copy -= p_copy[:, :, :, 0, None]
    return p_copy


def sc_hypo_batch(p_ref, p, njoints=16):
    # Only perform the scale correction
    p_ref, p = p_ref.reshape((-1, 3, njoints+1)), p.reshape((-1, 3, njoints+1))

    #p_ref = p_ref.clone() - p_ref[:, :, 0, None]
    #p = p.clone() - p[:, :, 0, None]

    scale_p = p.reshape(-1, 3*njoints+3).norm(p=2, dim=1, keepdim=True)
    scale_p_ref = p_ref.reshape(-1, 3*njoints+3).norm(p=2, dim=1, keepdim=True)
    scale = scale_p_ref / scale_p
    print(scale.min(), scale.max(), p.shape, scale.shape)
    p = (p.reshape(-1, 3*njoints+3) * scale).reshape(-1, 3, njoints+1)
    return p

    err = (p - p_ref).norm(p=2, dim=1).mean(dim=1)
    return err

    mu_gt = p_ref.mean(dim=2)
    mu_pred = p.mean(dim=2)

    X0 = p_ref - mu_gt[:, :, None]
    Y0 = p - mu_pred[:, :, None]

    ssX = (X0 ** 2.).sum(dim=(1, 2))
    ssY = (Y0 ** 2.).sum(dim=(1, 2))

    # centred Frobenius norm
    normX = torch.sqrt(ssX) + 1e-6
    normY = torch.sqrt(ssY) + 1e-6

    Y0 = Y0 * (normX / normY)[:, None, None]
    Y0 = Y0 - Y0[:, :, 0, None]

    return Y0


def compute_3DPCK(poses_gt, poses_pred, threshold=150):
    # poses_pred.shape (bs, 3, 17) or (n_hypo, bs, 3, 17)
    assert len(poses_pred.shape) == 3 or len(poses_pred.shape) == 4
    # see https://arxiv.org/pdf/1611.09813.pdf
    joints_to_use = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]
    if len(poses_pred.shape) == 3:
        # compute distances to gt:
        distances = torch.sqrt(torch.sum((poses_gt[:, :, joints_to_use]
                                          - poses_pred[:, :, joints_to_use])**2, dim=1))
        pck = torch.count_nonzero(distances < threshold, dim=1) / len(joints_to_use)
    else:
        distances = torch.sqrt(torch.sum((poses_gt[:, :, :, joints_to_use]
                                          - poses_pred[:, :, :, joints_to_use]) ** 2, dim=2))
        pck = torch.count_nonzero(distances < threshold, dim=2) / len(joints_to_use)
    return pck


def compute_3DPCK_list(poses_gt, poses_pred, min_th=1, max_th=300, step=1):
    # computes Correct Poses Score (CPS) (https://arxiv.org/abs/2011.14679)
    # for different thresholds
    # poses_pred.shape (bs, 3, 17) or (n_hypo, bs, 3, 17)
    assert len(poses_pred.shape) == 3 or len(poses_pred.shape) == 4
    thresholds = torch.arange(min_th, max_th+1, step).tolist()

    if len(poses_pred.shape) == 3:
        cp_values = torch.empty((poses_pred.shape[0], len(thresholds)), dtype=torch.double)
        for i, threshold in enumerate(thresholds):
            cp_values[:, i] = compute_CP(poses_gt, poses_pred, threshold=threshold)
    else:
        cp_values = torch.empty((poses_pred.shape[0], poses_pred.shape[1], len(thresholds)), dtype=torch.double)
        for i, threshold in enumerate(thresholds):
            cp_values[:, :, i] = compute_CP(poses_gt, poses_pred, threshold=threshold)
    return cp_values


def compute_CP(poses_gt, poses_pred, threshold=180):
    # poses_pred.shape (bs, 3, 17) or (bs, n_hypo, 3, 17)
    assert len(poses_pred.shape) == 3 or len(poses_pred.shape) == 4
    joints_to_use = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    if len(poses_pred.shape) == 3:
        # compute distances to gt:
        distances = torch.sqrt(torch.sum((poses_gt[:, :, joints_to_use]
                                          - poses_pred[:, :, joints_to_use])**2, dim=1))
        correct_poses = torch.count_nonzero(distances < threshold, dim=1) == len(joints_to_use)
    else:
        distances = torch.sqrt(torch.sum((poses_gt[:, :, :, joints_to_use]
                                          - poses_pred[:, :, :, joints_to_use]) ** 2, dim=2))
        # distances.shape (n_hypo, bs, 16)
        # pose is correct if all joints of a pose have distance < threshold
        correct_poses = torch.count_nonzero(distances < threshold, dim=2) == len(joints_to_use)
    return correct_poses


def compute_CP_list(poses_gt, poses_pred, min_th=1, max_th=300, step=1):
    # computes Correct Poses Score (CPS) (https://arxiv.org/abs/2011.14679)
    # for different thresholds
    # poses_pred.shape (bs, 3, 17) or (n_hypo, bs, 3, 17)
    assert len(poses_pred.shape) == 3 or len(poses_pred.shape) == 4
    thresholds = torch.arange(min_th, max_th+1, step).tolist()

    if len(poses_pred.shape) == 3:
        cp_values = torch.empty((poses_pred.shape[0], len(thresholds)), dtype=torch.double)
        for i, threshold in enumerate(thresholds):
            cp_values[:, i] = compute_CP(poses_gt, poses_pred, threshold=threshold)
    else:
        cp_values = torch.empty((poses_pred.shape[0], poses_pred.shape[1], len(thresholds)), dtype=torch.double)
        for i, threshold in enumerate(thresholds):
            cp_values[:, :, i] = compute_CP(poses_gt, poses_pred, threshold=threshold)
    return cp_values