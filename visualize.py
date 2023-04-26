from utils.plot_real17j_scaled import plot_real17j_1f_scaled
import torch
import pickle

sampled_poses = pickle.load(open("res/samples3d.pkl", "rb"))

with torch.no_grad():
    plot_real17j_1f_scaled(sampled_poses[0].reshape(1, 51).cpu().numpy(), sampled_poses[0].reshape(51).cpu().numpy())

poses = sampled_poses.cpu().numpy()
pickle.dump(poses, open("res/samples3d_np.pkl", "wb"))


print('done')
