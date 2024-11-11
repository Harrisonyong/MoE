#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   moe.py
@Time    :   2024/11/01 10:38:22
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   None
"""
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
import torch.nn as nn

MOE_TOP_K = 2
CONSTANT = 2


class Expert(Module):
    """
    MLP for MoE
    """

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # 上采样
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)

        self.act_fn = nn.SiLU

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class CopyExpert(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_ids):
        return input_ids


class ZeroExpert(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_ids):
        return torch.zeros_like(input_ids).to(input_ids.device)


# TODO: 将expert进行封装，核心创新点是通过一个MLP来控制
class ConstantExpert(Module):
    def __init__(self, expert: Expert):
        super().__init__()
        self.constant = torch.nn.Parameter(torch.empty((expert.hidden_size)))
        torch.nn.init.normal_(self.constant)

        self.wg = torch.nn.Linear(expert.hidden_size, 2, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, inputs):
        weight = self.softmax(self.wg(inputs))
        return torch.einsum(
            "b,bd->bd", [weight[:, 0].type_as(inputs), inputs]
        ) + torch.einsum("b,bd->bd", [weight[:, 1].type_as(inputs), self.constant])


class Router(Module):
    def __init__(self, model_dim: int, num_experts: int, is_gate_mlp: bool):
        super().__init__()
        self.num_experts = num_experts
        if is_gate_mlp:
            # TODO: 相比于传统MOE增加了一个MLP
            self.gate_layer = torch.nn.Sequential(
                torch.nn.Linear(model_dim, num_experts * 8, bias=False).float(),
                torch.nn.Tanh(),
                torch.nn.Linear(num_experts * 8, num_experts, bias=False).float(),
            ).float()
        else:
            self.gate_layer = torch.nn.Linear(
                model_dim, num_experts, bias=False
            ).float()

        self.pre_gate = torch.nn.Linear(num_experts, num_experts, bias=False)

    def forward(self, inputs: Tensor, pre_router_residual=None):
        inputs = inputs.float()

        # (batch * sequence_length, n_experts)
        router_logits = self.gate_layer(inputs)

        if pre_router_residual is not None:
            router_logits += self.pre_gate(
                pre_router_residual.to(self.pre_gate.weight.dtype)
            )

        # (batch * sequence_length, n_experts)
        router_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        # (batch * sequence_length, n_selected_experts)
        selected_router_weight, selected_expert_idx = torch.topk(
            router_weights, k=MOE_TOP_K, dim=1
        )

        # TODO: 文章创新点，如果topk的idx是num_experts-1，则置零
        selected_router_weight = torch.where(
            selected_expert_idx == self.num_experts - 1,
            torch.zeros_like(selected_router_weight)
            .to(selected_router_weight.dtype)
            .to(selected_router_weight.device),
            selected_router_weight,
        )
        # (batch * sequence_length, n_selected_experts)
        selected_router_weight /= selected_router_weight.sum(dim=-1, keepdim=True)

        # expert_info: 记录选择专家的token和权重
        expert_info = dict()
        for expert_id in range(self.num_experts):
            token_ids, weight_ids = torch.where(selected_expert_idx == expert_id)
            expert_info[expert_id] = [
                token_ids,
                selected_router_weight[token_ids, weight_ids],
            ]
        return expert_info, router_logits


class MOELayer(Module):
    def __init__(
        self,
        hidden_size,
        expert_intermediate_size,
        num_experts,
        is_gate_mlp=True,
    ):
        super().__init__()
        self.num_experts = num_experts
        expert = Expert(hidden_size, expert_intermediate_size)
        self.experts = torch.nn.ModuleList(
            [CopyExpert(), ZeroExpert()]
            + [ConstantExpert(expert) for _ in range(CONSTANT)]
            + [expert for _ in range(num_experts - CONSTANT - 2)]
        )
        self.router = Router(hidden_size, num_experts, is_gate_mlp)

    def forward(self, hidden_sates: Tensor, pre_router_residual=None):
        batch_size, seq_length, hidden_dim = hidden_sates.shape
        hidden_sates = hidden_sates.view(-1, hidden_dim)
        expert_info, router_logits = self.router(hidden_sates, pre_router_residual)
        output = torch.zeros(
            (batch_size * seq_length, hidden_dim),
            dtype=hidden_sates.dtype,
            device=hidden_sates.device,
        )
        for expert, token_ids_and_weight in expert_info.items():
            token_ids, weight = token_ids_and_weight
            weight = weight.unsqueeze(-1)
            tokens = hidden_sates.index_select(0, token_ids)
            expert_output = self.experts[expert](tokens)
            expert_output *= weight
            output.index_add_(0, token_ids, expert_output)
        output = output.reshape(batch_size, seq_length, hidden_dim)
        return output, router_logits
