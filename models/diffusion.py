import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from einops import rearrange, reduce, repeat

from models.modules.beta_schedules import linear_beta_schedule, cosine_beta_schedule
from models.modules.utils import extract, default
from typing import Literal


_LOSSTYPE = Literal["l1", "l2"]
_OBJECTIVE = Literal["pred_noise", "pred_x0"]
_BETASCHEDULE = Literal["cosine", "linear"]


class GaussianDiffusion(nn.Module):
    """
    Gaussian diffusion model:
     modified from https://github.com/lucidrains/denoising-diffusion-pytorch
    """
    def __init__(
        self,
        denoise_fn: nn.Module,
        condition_emb: nn.Module,
        *,
        timesteps: int = 25,
        loss_type: _LOSSTYPE = 'l2',
        objective: _OBJECTIVE = 'pred_noise',
        beta_schedule: _BETASCHEDULE = 'cosine',
        p2_loss_weight_gamma: float = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k: int = 1,
        scaling_3d_pose: float = 10.,
        noise_scale: float = 1.,
        num_joints: int = 16,
        **kwargs
    ):
        super().__init__()

        self.num_joints = num_joints
        self.denoise_fn = denoise_fn
        self.objective = objective
        self.condition_emb = condition_emb
        self.scaling_3d_pose = scaling_3d_pose
        self.noise_scale = noise_scale

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps, s=getattr(kwargs, 'cosine_offset', 0.008))
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), dtype=torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond, clip_denoised: bool):
        model_output = self.denoise_fn(x, t, cond)

        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t=t, noise=model_output)
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if clip_denoised:
            """Clips the x values to a reasonable value at each step during inference, probably not necessary"""
            x_start.clamp_(-3. * self.scaling_3d_pose, 3. * self.scaling_3d_pose)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=True, sample_mean_only=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) * self.noise_scale

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        if not sample_mean_only:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        else:
            return model_mean

    @torch.no_grad()
    def p_sample_loop(
            self, cond, shape,
            sample_mean_only=False, print_progress=False
    ):
        device = self.betas.device

        b = shape[0]
        noise = torch.randn(shape, device=device) * self.noise_scale

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps, disable=not print_progress):
            noise = self.p_sample(
                noise, torch.full((b,), i, device=device, dtype=torch.long),
                cond, sample_mean_only=sample_mean_only
            )

        return noise

    @torch.no_grad()
    def sample(
            self, cond, batch_size=16, sample_mean_only=False, n_hypotheses_to_sample=1,
            print_progress=False
    ):
        if getattr(self.condition_emb, 'sampleable', False) and not sample_mean_only:
            cond_emb = self.condition_emb(cond, num_samples_to_draw=n_hypotheses_to_sample)
            cond_emb = rearrange(cond_emb, 'b n_hypo f -> (b n_hypo) f', b=cond.shape[0], n_hypo=n_hypotheses_to_sample)
        else:
            cond_emb = self.condition_emb(cond, sample_mean_only=sample_mean_only)
            cond_emb = repeat(cond_emb, 'b 1 f -> (b n_hypo) f', n_hypo=n_hypotheses_to_sample)

        return self.p_sample_loop(
            cond_emb, (cond_emb.shape[0], 3*self.num_joints),
            sample_mean_only=sample_mean_only,
            print_progress=print_progress
        ) / self.scaling_3d_pose

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        raise NotImplementedError

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start) * self.noise_scale)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, cond, noise = None):
        b = x_start.shape[0]
        noise = default(noise, lambda: torch.randn_like(x_start) * self.noise_scale)

        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.denoise_fn(x, t, cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target.expand_as(model_out), reduction='none')

        if len(loss.shape) == 3:
            # Probably not necessary since the output dim is already 3
            loss = reduce(loss, 'b s ... -> b s (...)', 'mean')
            # A simple regularization of the output to make all poses realistic in magnitude
            loss = loss.min(dim=1)[0] + 0.01*loss.mean(dim=1)
        else:
            loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        return loss.mean()

    def forward(self, img, cond, *args, **kwargs):
        b, j, device, = *img.shape, img.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        cond_emb = self.condition_emb(cond)
        cond_emb = rearrange(cond_emb, 'b n_hypo f -> (b n_hypo) f', b=cond.shape[0], n_hypo=1)

        img = img * self.scaling_3d_pose

        #img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, cond_emb, *args, **kwargs)