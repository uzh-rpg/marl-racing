import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
import torch
from torch import nn


class PerceiverI(nn.Module):
    def __init__(self, kv_dim, num_queries, num_heads=4, dim_head=32):
        super().__init__()
        self.num_heads = num_heads
        query_dim = dim_head * self.num_heads
        self.latents = nn.Parameter(torch.randn(num_queries, query_dim))
        self.kv_layers = nn.Sequential(
            nn.Linear(kv_dim, query_dim), nn.LayerNorm(query_dim)
        )
        self.multihead_attention = nn.MultiheadAttention(
            query_dim, self.num_heads, batch_first=True, kdim=query_dim, vdim=query_dim
        )

    def forward(self, inputs, key_padding_mask):
        b = inputs.shape[0]
        inputs = self.kv_layers(inputs.flatten(0, 1)).unflatten(0, [b, -1])
        batch_latent = torch.tile(self.latents[None, :], (b, 1, 1))
        x_attn = self.multihead_attention(
            query=batch_latent,
            key=inputs,
            value=inputs,
            key_padding_mask=key_padding_mask,
        )[0]
        return x_attn


class CustomPerceiver(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        model_cfg: dict = None,
        encoder_kwargs: dict = None,
    ):
        super().__init__(observation_space, features_dim)
        self.model_cfg = model_cfg
        self.num_queries = model_cfg["perceiver"]["num_queries"]
        self.num_heads = model_cfg["perceiver"]["num_heads"]
        self.dim_head = model_cfg["perceiver"]["dim_head"]
        self.variable_flattened_dim = self.num_queries * self.num_heads * self.dim_head
        self.obs_dim_variable = encoder_kwargs["obs_dim_variable"]
        self.obs_dim_fixed = encoder_kwargs["obs_dim_fixed"]
        self.variable_feature_dim = encoder_kwargs["variable_feature_dim"]
        self.variable_encoder = PerceiverI(
            encoder_kwargs["variable_feature_dim"],
            self.num_queries,
            num_heads=self.num_heads,
            dim_head=self.dim_head,
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        features = observations
        b = features.shape[0]
        variable_features = features[:, self.obs_dim_fixed :].reshape(
            [b, -1, self.variable_feature_dim]
        )
        valid_mask = torch.abs(variable_features.sum([1, 2])) != 0
        out_attn_features = torch.zeros(
            [b, self.variable_flattened_dim], device=features.device
        )
        if valid_mask.sum() != 0:
            key_padding_mask = (
                torch.abs(variable_features[valid_mask, :, :].sum(-1)) == 0
            )
            out_attn_features[valid_mask] = self.variable_encoder.forward(
                variable_features[valid_mask, :, :], key_padding_mask
            ).flatten(1, 2)
        return torch.cat([features[:, : self.obs_dim_fixed], out_attn_features], dim=1)
