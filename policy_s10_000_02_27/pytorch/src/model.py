# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/Emerge-Lab/gpudrive
# License: MIT
"""
Standalone FFN policy network matching the gpudrive `late_fusion.NeuralNet`
architecture. Reimplemented here to avoid pulling in the full gpudrive runtime
(madrona_gpudrive, pufferlib, box) while still loading published weights via
`PyTorchModelHubMixin`.
"""
import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

EGO_FEAT_DIM = 6
PARTNER_FEAT_DIM = 6
ROAD_GRAPH_FEAT_DIM = 13
TOP_K_ROAD_POINTS = 200


class NeuralNet(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/Emerge-Lab/gpudrive",
    docs_url="https://arxiv.org/abs/2502.14706",
    tags=["ffn"],
):
    """FFN late-fusion actor-critic policy for the gpudrive environment."""

    def __init__(
        self,
        action_dim=91,
        input_dim=64,
        hidden_dim=128,
        dropout=0.0,
        act_func="tanh",
        max_controlled_agents=64,
        obs_dim=2984,
        config=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.max_controlled_agents = max_controlled_agents
        self.max_observable_agents = max_controlled_agents - 1
        self.obs_dim = obs_dim
        self.num_modes = 3
        self.dropout = dropout
        self.act_func = nn.Tanh() if act_func == "tanh" else nn.GELU()

        self.ego_state_idx = EGO_FEAT_DIM
        self.partner_obs_idx = PARTNER_FEAT_DIM * self.max_controlled_agents

        self.ego_embed = nn.Sequential(
            nn.Linear(self.ego_state_idx, input_dim),
            nn.LayerNorm(input_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            nn.Linear(input_dim, input_dim),
        )
        self.partner_embed = nn.Sequential(
            nn.Linear(PARTNER_FEAT_DIM, input_dim),
            nn.LayerNorm(input_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            nn.Linear(input_dim, input_dim),
        )
        self.road_map_embed = nn.Sequential(
            nn.Linear(ROAD_GRAPH_FEAT_DIM, input_dim),
            nn.LayerNorm(input_dim),
            self.act_func,
            nn.Dropout(self.dropout),
            nn.Linear(input_dim, input_dim),
        )
        self.shared_embed = nn.Sequential(
            nn.Linear(self.input_dim * self.num_modes, self.hidden_dim),
            nn.Dropout(self.dropout),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def unpack_obs(self, obs_flat):
        ego_state = obs_flat[:, : self.ego_state_idx]
        partner_obs = obs_flat[:, self.ego_state_idx : self.partner_obs_idx]
        roadgraph_obs = obs_flat[:, self.partner_obs_idx :]
        road_objects = partner_obs.view(
            -1, self.max_observable_agents, PARTNER_FEAT_DIM
        )
        road_graph = roadgraph_obs.view(
            -1, TOP_K_ROAD_POINTS, ROAD_GRAPH_FEAT_DIM
        )
        return ego_state, road_objects, road_graph

    def encode_observations(self, observation):
        ego_state, road_objects, road_graph = self.unpack_obs(observation)
        ego_embed = self.ego_embed(ego_state)
        partner_embed, _ = self.partner_embed(road_objects).max(dim=1)
        road_map_embed, _ = self.road_map_embed(road_graph).max(dim=1)
        embed = torch.cat([ego_embed, partner_embed, road_map_embed], dim=1)
        return self.shared_embed(embed)

    def forward(self, obs):
        """Return (action_logits, value) for a batch of flattened observations."""
        hidden = self.encode_observations(obs)
        logits = self.actor(hidden)
        value = self.critic(hidden)
        return logits, value
