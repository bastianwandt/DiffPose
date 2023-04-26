import torch
import tables as tb
import numpy as np
import scipy.optimize as opt
from viz.plot_p3d import plot17j, plot17j_multi, swap_pose_axes17j, BONES17j, get_limits


def gauss2d_angle(xy, amp, x0, y0, sigma_x, theta, sigma_y):
    x, y = xy
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    out = amp * np.exp(- (a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0)
                          + c * ((y - y0) ** 2)))
    return out.ravel()


def fit_single_gaussian(esti_hm):
    # esti_hm.shape (16, 64, 64)
    # p2d.shape (32,)  x0x1x2...y0y1y2
    n_joints = esti_hm.shape[0]
    solution = []
    for j in range(esti_hm.shape[0]):  # iterate over each joint heatmap
        single_hm = esti_hm[j]

        p_guess = np.unravel_index(single_hm.argmax(), single_hm.shape)

        w_guess = p_guess[1]
        h_guess = p_guess[0]

        max_confi = single_hm.max().item()
        gt_sigma = 2.
        hm_w = hm_h = 64
        # initial guess:
        guess = [max_confi, w_guess, h_guess, gt_sigma, 0, gt_sigma]

        x = np.linspace(0, hm_w-1, hm_w)
        y = np.linspace(0, hm_h-1, hm_h)
        x, y = np.meshgrid(x, y)

        # first input is function to fit, second is array of x and y values (coordinates) and third is heatmap array
        try:
            pred_params, uncert_cov = opt.curve_fit(gauss2d_angle, (x, y), single_hm.flatten(), p0=guess)
        except:
            print("Runtime error in curve fitting")
            pred_params = guess
            data_fitted = gauss2d_angle((x, y), *pred_params).reshape(hm_w, hm_h)
            rmse_fit = np.sqrt(np.mean((single_hm - data_fitted) ** 2))
            print(rmse_fit)

        data_fitted = gauss2d_angle((x, y), *pred_params).reshape(hm_w, hm_h)
        rmse_fit = np.sqrt(np.mean((single_hm - data_fitted) ** 2))

        solution.extend(pred_params)
        solution.append(rmse_fit)
    # returns a single list of fit parameters:
    # A_0, mux_0, muy_0, sigmax_0, theta_0, sigmay_0, fiterr_0, A_1, ...
    return solution


def get_max_uncertainty_and_filename(h5_file, threshold=0.4, early_stop=False):
    data = tb.open_file(h5_file)
    datasplit = data.root.dataset.testset
    frame_info = []

    hard_idx = 0
    x_std = []
    y_std = []
    for i, sample in enumerate(datasplit.iterrows()):
        if not sample['hardsubset']:
            #print(max(percentage), sample['hardsubset'])
            continue
        elif hard_idx % 10 != 0:
            hard_idx += 1
            continue

        hm = sample['heatmap']
        print('hard samples ', hard_idx)
        gauss_fits = np.asarray(fit_single_gaussian(hm)).reshape(-1, 7)

        if sample['subaction'] == b'1':
            filename = '{}/Videos/{} 1.{}.mp4'.format(
                sample['subject'].decode("utf-8"),
                sample['action'].decode("utf-8"),
                sample['cam'].decode("utf-8")
            )
        else:
            filename = '{}/Videos/{}.{}.mp4'.format(
                sample['subject'].decode("utf-8"),
                sample['action'].decode("utf-8"),
                sample['cam'].decode("utf-8")
            )
        frame_idx = sample['frame_orig']
        frame_info.append([
            filename, frame_idx,
            # sample['action'], sample['subject'],
            # sample['subaction'], sample['cam'],
            # sample['frame_orig'],
            gauss_fits[:, 3].max(),
            gauss_fits[:, 5].max(),
            i, hard_idx,
            sample['gt_3d']
        ])
        x_std.append(gauss_fits[:, 3].max())
        y_std.append(gauss_fits[:, 5].max())
        hard_idx += 1
        if hard_idx > 100 and early_stop:
            break

    x_ind = np.argpartition(x_std, -10)[-10:]
    y_ind = np.argpartition(y_std, -10)[-10:]

    for x in x_ind:
        print(frame_info[x][:-1])
        gt = frame_info[x][-1]
        gt[:17] -= gt[0]
        gt[17:34] -= gt[17]
        gt[34:] -= gt[34]
        gt[17:] *= -1
        plot17j(gt)
    for x in y_ind:
        print(frame_info[x][:-1])

    print(max(x_std), max(y_std))


if __name__=='__main__':
    get_max_uncertainty_and_filename('/media/karl/Samsung_T5/data/h36m/heatmaps_corrected.h5')