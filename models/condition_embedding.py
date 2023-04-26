import torch
import torch.nn as nn
import math
from einops import rearrange, repeat, reduce
from typing import Literal


_NORMALIZATION = Literal["image_half", "image", "mean_std", "none"]


class HeatmapSampler(nn.Module):

    def __init__(
            self, min_threshold: float = -1., temperature_scaling: float = 1.,
            return_ordered: bool = True, normalize_poses: bool = True,
            normalization_method: _NORMALIZATION = 'image_half',
            sample_with_replacement: bool = True
    ) -> object:
        """

        Parameters
        ----------
        min_threshold
        temperature_scaling
        return_ordered
        normalize_poses
        normalization_method
        sample_with_replacement
        """
        super().__init__()
        self.min_threshold = min_threshold
        self.tau = temperature_scaling
        self.return_ordered = return_ordered
        self.normalize_poses = normalize_poses
        self.normalization_method = normalization_method
        self.sample_with_replacement = sample_with_replacement

    def normalize(self, x, y, heatmap_size=64):
        if self.normalization_method == 'mean_std_per_sample':
            # Normalize the poses based on mean and std
            x = x - reduce(x, 'b j n -> b () n', reduction='mean')
            y = y - reduce(y, 'b j n -> b () n', reduction='mean')

            xy = rearrange([x, y], 'd b j n -> b j n d')
            std = reduce(xy, 'b j n d -> b () n', reduction=torch.std)

            x = x / std
            y = y / std

        elif self.normalization_method == 'mean_std':
            # Normalize the poses based on mean and std over all samples
            x = x - reduce(x, 'b j n -> b () ()', reduction='mean')
            y = y - reduce(y, 'b j n -> b () ()', reduction='mean')

            xy = rearrange([x, y], 'd b j n -> b j n d')
            std = reduce(xy, 'b j n d -> b () ()', reduction=torch.std)

            x = x / std
            y = y / std

        elif self.normalization_method == 'image_half':
            # Normalize the poses to a fixed interval
            x = (x - 0.5 * heatmap_size) / heatmap_size
            y = (y - 0.5 * heatmap_size) / heatmap_size
        else:
            # Normalize the poses to a fixed interval
            x = 2 * (x - 0.5 * heatmap_size) / heatmap_size
            y = 2 * (y - 0.5 * heatmap_size) / heatmap_size

        return x, y

    def sample(self, heatmap, num_samples=16, return_probs=False):
        heatmap_probs = heatmap.clone()

        # Enforce unlikely position to have zero probability
        if self.min_threshold < 0:
            heatmap_probs[heatmap_probs < 0] = 0
        else:
            heatmap_probs[heatmap_probs < self.min_threshold] = 0.
        # Apply sharpening of scores
        heatmap_probs = heatmap_probs ** self.tau

        # Normalize to get a probability over all positions
        b, j, _, _ = heatmap_probs.shape
        heatmap_probs = rearrange(heatmap_probs, 'b j w h -> (b j) (w h)')

        # Sample one position per joint and batch
        samples = torch.multinomial(heatmap_probs, num_samples, replacement=self.sample_with_replacement)
        if self.return_ordered or return_probs:  # Sorts the samples in decreasing likelihood order
            sample_probs = torch.gather(heatmap_probs, dim=-1, index=samples)
            if self.return_ordered:
                ordering = torch.argsort(sample_probs, dim=-1, descending=True)
                samples = torch.gather(samples, dim=-1, index=ordering)
                sample_probs = torch.gather(sample_probs, dim=-1, index=ordering)

            sample_probs = rearrange(sample_probs, '(b j) n -> b j n', b=b, j=j, n=num_samples)

        samples = rearrange(samples, '(b j) n -> b j n', b=b, j=j, n=num_samples).float()

        # Unravel index
        sampled_joint_x = (samples % heatmap.shape[-2]).float()
        sampled_joint_y = torch.div(samples, heatmap.shape[-2], rounding_mode='floor').float()
        if self.normalize_poses:
            sampled_joint_x, sampled_joint_y = self.normalize(sampled_joint_x, sampled_joint_y, heatmap.shape[-1])

        sampled_joint = torch.stack((sampled_joint_x, sampled_joint_y), dim=-2)
        sampled_joint = rearrange(sampled_joint, 'b j d n -> b n (j d)', b=b, j=j, n=num_samples, d=2)

        if not return_probs:
            return sampled_joint  # Shape: B x N_samples x 2*N_joints
        else:
            sample_probs = rearrange(sample_probs, 'b j n -> b n j', b=b, j=j, n=num_samples)
            return sampled_joint, sample_probs

    def get_maximum(self, heatmap, return_probs=False):
        b, j, _, _ = heatmap.shape

        probs, samples = torch.max(rearrange(heatmap, 'b j w h -> b j (w h)', b=b, j=j), dim=-1, keepdim=True)

        # Unravel index
        sampled_joint_x = (samples % heatmap.shape[-2]).float()
        sampled_joint_y = torch.div(samples, heatmap.shape[-2], rounding_mode='floor').float()

        if self.normalize_poses:
            sampled_joint_x, sampled_joint_y = self.normalize(sampled_joint_x, sampled_joint_y, heatmap.shape[-1])

        sampled_joint = torch.stack((sampled_joint_x, sampled_joint_y), dim=-2)
        sampled_joint = rearrange(sampled_joint, 'b j d n -> b n (j d)', b=b, j=j, n=1, d=2)

        if not return_probs:
            return sampled_joint  # Shape: B x N_samples x 2*N_joints
        else:
            probs = rearrange(probs, 'b j n -> b n j', b=b, j=j, n=1)
            return sampled_joint, probs


class HeatmapSamplingCat(nn.Module):
    def __init__(
            self, min_threshold=-1,
            temperature_scaling=1.,
            num_samples=16, sort_samples=True,
            concat_likelihood=False,
            normalization_method='mean_std'
    ):
        super().__init__()
        self.sampleable = False
        self.concat_likelihood = concat_likelihood
        self.num_samples = num_samples - 1
        self.sampler = HeatmapSampler(
            min_threshold,
            temperature_scaling,
            return_ordered=sort_samples,
            normalization_method=normalization_method
        )

    def forward(self, heatmap, num_samples_to_draw=1, **kwargs):
        b = heatmap.shape[0]
        heatmap = repeat(
            heatmap, 'batch joints width height -> (batch n_samples) joints width height',
            n_samples=num_samples_to_draw
        )

        if self.num_samples > 0:
            if not self.concat_likelihood:
                random_samples = self.sampler.sample(heatmap, self.num_samples)
                max_sample = self.sampler.get_maximum(heatmap)
            else:
                random_samples, sample_probs = self.sampler.sample(heatmap, self.num_samples, return_probs=True)
                max_sample, max_prob = self.sampler.get_maximum(heatmap, return_probs=True)
                probs = torch.cat((max_prob, sample_probs), dim=1)

            samples = torch.cat((max_sample, random_samples), dim=1)
        else:
            if not self.concat_likelihood:
                samples = self.sampler.get_maximum(heatmap)
            else:
                samples, probs = self.sampler.get_maximum(heatmap, return_probs=True)

        if self.concat_likelihood:
            samples = torch.cat((
                rearrange(samples, 'b p (j d) -> b p j d', j=16, d=2),
                rearrange(probs, 'b p (j d) -> b p j d', j=16, d=1)
            ), dim=-1)
            samples = rearrange(
                samples, '(b n_samples) p j d -> b n_samples (p j d)',
                b=b, n_samples=num_samples_to_draw, p=self.num_samples+1, j=16, d=3
            )
        else:
            samples = rearrange(
                samples, '(b n_samples) p (j d) -> b n_samples (p j d)',
                b=b, n_samples=num_samples_to_draw, p=self.num_samples+1, j=16, d=2
            )

        return samples


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, base_freq=10000):
        super().__init__()
        self.dim = dim
        self.base_freq = base_freq

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.base_freq) / (half_dim - 1)  # Natural logarithm
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


class ChannelPosEmb(nn.Module):
    def __init__(self, start=-1., end=1., num_steps=32):
        super().__init__()

        self.register_buffer('centers', torch.linspace(start, end, num_steps))
        self.frequency = 0.25 * math.pi * (num_steps / (end - start))

    def forward(self, x):
        dists = x[:, None] - self.centers[None, :]

        emb = torch.cos(dists * self.frequency) ** 2
        emb[torch.abs(dists * self.frequency) > 0.5 * math.pi] = 0
        return emb


class HeatmapSamplingPoseformer(nn.Module):
    def __init__(self, num_samples=16, condition_dim=32, embedding_dim=64,
                 normalization_method='mean_std', dropout_joint_method=None,
                 dropout_prob=0.01, uncertain_thresh=1.0,
                 sample_with_replacement=True,
                 sampleable=False,
                 positional_embedding='sin',
                 old_behaviour=True,
                 project_positional_embedding=False,
                 without_transformer=False,
                 without_argmax=False,
                 without_posemb_scaling=False,
                 use_full_heatmap=False
                 ):
        super().__init__()
        if not without_argmax:
            self.num_samples = num_samples - 1
        else:
            self.num_samples = num_samples
        self.emb_dim = embedding_dim
        self.dropout_method = dropout_joint_method
        self.dropout_prob = dropout_prob
        self.uncertain_thresh = uncertain_thresh
        self.sample_with_replacement = sample_with_replacement
        self.positional_embedding = positional_embedding
        self.without_transformer = without_transformer
        self.without_argmax = without_argmax
        self.without_posemb_scaling = without_posemb_scaling
        self.use_full_heatmap = use_full_heatmap

        if project_positional_embedding:
            self.pos_proj = nn.Linear(2*self.emb_dim, 2*self.emb_dim)
        else:
            self.pos_proj = None

        if self.positional_embedding == 'sin':
            self.pos_encoder = SinusoidalPosEmb(self.emb_dim)
        elif self.positional_embedding == 'channel':
            self.pos_encoder = ChannelPosEmb(num_steps=self.emb_dim)
            normalization_method = 'image'
        else:
            self.pos_encoder = nn.Sequential(
                nn.Linear(2, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 2*self.emb_dim)
            )

        self.sampler = HeatmapSampler(
            return_ordered=False, normalization_method=normalization_method,
            sample_with_replacement=self.sample_with_replacement
        )

        self.joint_embedding = nn.Embedding(16, self.emb_dim*2)
           
        if not self.without_transformer:
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=2*self.emb_dim,  # 2D-joint position + likelihood of sample
                    nhead=4,
                    dim_feedforward=512,
                    batch_first=True
                ),
                num_layers=4,
            )
        else:
            self.transformer = None

        self.old = old_behaviour

        # Process
        self.conf_out = nn.Linear(2*self.emb_dim, 1)
        if self.old:
            self.pos_out = nn.Linear(2*self.emb_dim, condition_dim*16)
        else:
            self.pos_out = nn.Linear(2*self.emb_dim*16, condition_dim*16)
        self.sampleable = sampleable

    def get_samples(self, heatmap, num_samples_to_draw):
        if self.num_samples > 0:
            random_samples, sample_probs = self.sampler.sample(heatmap, self.num_samples * num_samples_to_draw,
                                                               return_probs=True)
            sample_probs = rearrange(sample_probs, 'b (n_hypo p) (j d) -> (b n_hypo) p (j d)', j=16, d=1,
                                     n_hypo=num_samples_to_draw, p=self.num_samples)
            random_samples = rearrange(random_samples, 'b (n_hypo p) (j d) -> (b n_hypo) p (j d)', j=16, d=2,
                                       n_hypo=num_samples_to_draw, p=self.num_samples)

            if not self.without_argmax:
                max_sample, max_prob = self.sampler.get_maximum(heatmap, return_probs=True)
                max_prob = repeat(max_prob, 'b 1 (j d) -> (b n_hypo) 1 (j d)', n_hypo=num_samples_to_draw, j=16, d=1)
                max_sample = repeat(max_sample, 'b 1 (j d) -> (b n_hypo) 1 (j d)', n_hypo=num_samples_to_draw, j=16,
                                    d=2)

                probs = torch.cat((max_prob, sample_probs), dim=1)
                samples = torch.cat((max_sample, random_samples), dim=1)
            else:
                probs = sample_probs
                samples = random_samples
        else:
            samples, probs = self.sampler.get_maximum(heatmap, return_probs=True)
            probs = repeat(probs, 'b 1 (j d) -> (b n_hypo) 1 (j d)', n_hypo=num_samples_to_draw, j=16, d=1)
            samples = repeat(samples, 'b 1 (j d) -> (b n_hypo) 1 (j d)', n_hypo=num_samples_to_draw, j=16, d=2)

        return samples, probs

    def get_full_heatmap(self, heatmap, num_samples_to_draw):
        x_pos = torch.linspace(-1, 1, heatmap.shape[-1], device=heatmap.device)
        y_pos = torch.linspace(-1, 1, heatmap.shape[-2], device=heatmap.device)

        x_pos, y_pos = torch.meshgrid((x_pos, y_pos))

        samples = rearrange([x_pos, y_pos], 'd w h -> (w h) d')
        samples = repeat(samples, 'n d -> (b n_hypo) n (j d)', b=heatmap.shape[0], j=16, n_hypo=num_samples_to_draw)

        probs = repeat(heatmap, 'b j w h -> (b n_hypo) (w h) (j d)', b=heatmap.shape[0], j=16, n_hypo=num_samples_to_draw, d=1)
        probs = torch.clamp_min(probs, min=0.)

        return samples, probs

    def forward(self, heatmap, num_samples_to_draw=1, **kwargs):
        b = heatmap.shape[0]

        if not self.use_full_heatmap:
            samples, probs = self.get_samples(heatmap, num_samples_to_draw)
        else:
            samples, probs = self.get_full_heatmap(heatmap, num_samples_to_draw)

        if self.positional_embedding == 'linear':
            positional_embedded_joints = rearrange(self.pos_encoder(rearrange(samples, 'b p (j d) -> b p j d', j=16, d=2)), 'b p j (d c) -> (b p j d) c', d=2)
        else:
            positional_embedded_joints = self.pos_encoder(rearrange(samples, 'b p (j d) -> (b p j d)', j=16, d=2))  # Embed the 2d positions

        probs = rearrange(probs, 'b p j -> (b j) p 1', j=16)
        positional_embedded_joints = rearrange(
            positional_embedded_joints,
            '(b p j d) c -> (b j) p (d c)', j=16, p=probs.shape[-2], d=2, c=self.emb_dim
        )
        if self.dropout_method == 'any' and self.training:
            # Randomly set the positional embedding to zero for a certain joint (all samples)
            dropout_mask = torch.rand(positional_embedded_joints.shape[0], device=positional_embedded_joints.device) > self.dropout_prob
            positional_embedded_joints = positional_embedded_joints * dropout_mask[:, None, None]
        elif self.dropout_method == 'uncertain' and self.training:
            # Randomly set the positional embedding to zero for a certain joint (all samples)
            #  Only applied to uncertain joints
            uncertain_mask = reduce(samples, 'b p (j d) -> (b j)', reduction=torch.std, d=2, j=16) < self.uncertain_thresh
            dropout_mask = torch.rand(positional_embedded_joints.shape[0],
                                      device=positional_embedded_joints.device) > self.dropout_prob
            dropout_mask = dropout_mask | uncertain_mask
            positional_embedded_joints = positional_embedded_joints * dropout_mask[:, None, None]

        if not self.without_posemb_scaling:
            positional_embedded_joints = positional_embedded_joints * probs  # Scale the positional embeddings based on the likelihood
        joint_embedding = self.joint_embedding(torch.arange(16, dtype=torch.long, device=probs.device))
        joint_embedding = repeat(joint_embedding, 'j c -> (b n_hypo) j c', b=b, n_hypo=num_samples_to_draw, j=16)

        # Sum the cosine embeddings for all samples (With optional projection layer)
        if self.pos_proj is not None:
            positional_embedded_joints = self.pos_proj(positional_embedded_joints)

        positional_embedded_joints = reduce(positional_embedded_joints, '(b j) p e -> b j e', reduction='sum', j=16)
        positional_embedded_joints = positional_embedded_joints + joint_embedding  # Add a positional embedding to the corresponding joint

        if self.transformer is not None:
            # positional_embedded_joints = rearrange(positional_embedded_joints, '(b j) p (d c) -> b (j p) (d c)', j=16, d=2, p=self.num_samples)
            positional_embedded_joints = self.transformer(positional_embedded_joints)  # Attend between the different samples

        # Merge the different joint embeddings into one pose vector
        if self.old:
            out = self.pos_out(positional_embedded_joints)
            out = reduce(out, 'b j e -> b e', reduction='sum')
        else:
            out = rearrange(positional_embedded_joints, 'b j e -> b (j e)')
            out = self.pos_out(out)
        out = rearrange(out, '(b n_hypo) e -> b n_hypo e', b=b)

        return out


class IdentityEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, c, **kwargs):
        c = rearrange(c, 'b f -> b 1 f')
        return c


def get_condition_embedding(cfg):
    if cfg.MODEL.CONDITION_TYPE == 'heatmap_sampling_concatenated':
        model = HeatmapSamplingCat(
            num_samples=cfg.MODEL.EXTRA.NUM_SAMPLES,
            sort_samples=cfg.MODEL.EXTRA.CONDITION_ORDERED_SAMPLES,
            concat_likelihood=getattr(cfg.MODEL.EXTRA, 'CONCAT_LIKELIHOOD', False),
            normalization_method=getattr(cfg.MODEL.EXTRA, 'NORM_METHOD', 'image_half')
        )
    elif cfg.MODEL.CONDITION_TYPE == 'embedded_poseformer':
        model = HeatmapSamplingPoseformer(
            num_samples=cfg.MODEL.EXTRA.NUM_SAMPLES,
            condition_dim=cfg.MODEL.CONDITION_DIM,
            normalization_method=getattr(cfg.MODEL.EXTRA, 'NORM_METHOD', 'mean_std'),
            dropout_joint_method=getattr(cfg.MODEL.EXTRA, 'DROP_METHOD', None),
            dropout_prob=getattr(cfg.MODEL.EXTRA, 'DROP_PROB', 0.1),
            sample_with_replacement=not getattr(cfg.MODEL.EXTRA, 'NO_REPLACEMENT', False),
            sampleable=getattr(cfg.MODEL.EXTRA, 'SAMPLEABLE', False),
            positional_embedding=getattr(cfg.MODEL.EXTRA, 'POS_EMB', 'sin'),
            old_behaviour=getattr(cfg.MODEL.EXTRA, 'OLD', True),
            project_positional_embedding=getattr(cfg.MODEL.EXTRA, 'PROJ_POS_EMB', False),
            without_transformer=getattr(cfg.MODEL.EXTRA, 'WITHOUT_TRANSFORMER', False),
            without_argmax=getattr(cfg.MODEL.EXTRA, 'WITHOUT_ARGMAX', False),
            without_posemb_scaling=getattr(cfg.MODEL.EXTRA, 'WITHOUT_POSEMB_SCALING', False),
            use_full_heatmap=getattr(cfg.MODEL.EXTRA, 'FULL_HEATMAP', False)
        )
    elif cfg.MODEL.CONDITION_TYPE == 'none' or cfg.MODEL.CONDITION_TYPE == 'pose_2d':
        model = IdentityEmbedding()
    else:
        raise RuntimeError("Unknown condition type: {}".format(cfg.MODEL.CONDITION_TYPE))
    return model


if __name__ == '__main__':
    m = HeatmapSamplingCat()
    x = (torch.rand(2, 16, 64, 64)).clamp_min(0.)
    y = m(x)

    print(x.shape, y.shape, y.max(), y.min())
    print(x.max(), x.min())
