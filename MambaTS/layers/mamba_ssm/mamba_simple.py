# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from tree_scan_lan import _C

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

# from layers.mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class _MST(Function):
    @staticmethod
    def forward(ctx, edge_index, edge_weight, vertex_index):
        edge_out = _C.mst_forward(edge_index, edge_weight, vertex_index)
        return edge_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return None, None, None


class _BFS(Function):
    @staticmethod
    def forward(ctx, edge_index, max_adj_per_vertex):
        sorted_index, sorted_parent, sorted_child, _ = _C.bfs_forward(
            edge_index, max_adj_per_vertex
        )
        return sorted_index, sorted_parent, sorted_child


class _Refine(Function):
    @staticmethod
    def forward(
        ctx, feature_in, edge_weight, sorted_index, sorted_parent, sorted_child
    ):
        feature_out = _C.tree_scan_refine_forward(
            feature_in, edge_weight, sorted_index, sorted_parent, sorted_child
        )

        ctx.save_for_backward(
            feature_out, edge_weight, sorted_index, sorted_parent, sorted_child
        )
        return feature_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        feature_out, edge_weight, sorted_index, sorted_parent, sorted_child = (
            ctx.saved_tensors
        )

        grad_feature, grad_edge = _C.tree_scan_refine_backward_feature(
            feature_out,
            edge_weight,
            sorted_index,
            sorted_parent,
            sorted_child,
            grad_output,
        )
        return grad_feature, grad_edge, None, None, None


def norm2_distance(fm_ref, fm_tar):
    diff = fm_ref - fm_tar
    weight = (diff * diff).sum(dim=-2)
    return torch.exp(weight)  # with - is for max tree


def cosine_distance(fm_ref, fm_tar):
    weight = -torch.cosine_similarity(fm_ref, fm_tar, dim=1)
    return torch.exp(weight)  # with - is for min tree


def batch_index_opr(data, index):
    with torch.no_grad():
        channel = data.shape[1]
        index = index.unsqueeze(1).expand(-1, channel, -1).long()
    data = torch.gather(data, 2, index)
    return data


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_casual_conv=True,
        use_fast_path=False,  # Fused kernel options
        layer_idx=None,
        dropout=0.0,
        n_vars=0,
        VPT_mode=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.use_casual_conv = use_casual_conv
        self.layer_idx = layer_idx
        self.n_vars = n_vars
        self.dropout = dropout
        # default: d_model -> d_model * 4
        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )

        # if self.use_fast_path and not self.use_casual_conv:
        #     raise NotImplementedError("When use_fast_path is True, use_casual_conv should be True.")
        if self.use_casual_conv:
            print("use casual_conv!")
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

        self.activation = "silu"
        self.act = nn.SiLU()

        if self.dropout > 0 and not self.use_fast_path:
            self.x_proj = nn.Sequential(
                nn.Linear(
                    self.d_inner,
                    self.dt_rank + self.d_state * 2,
                    bias=False,
                    **factory_kwargs,
                ),
                # Dropout here!
                nn.Dropout(dropout),
            )
        else:
            self.x_proj = nn.Linear(
                self.d_inner,
                self.dt_rank + self.d_state * 2,
                bias=False,
                **factory_kwargs,
            )

        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(
        self, input_states, inference_params=None, ids_restore=None, context_len=None
    ):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        device = input_states.device
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(input_states).transpose(
            1, 2
        )  # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, dim=1)

        # hidden_states = self.act(
        #    self.conv1d(hidden_states)[..., :seq_len]
        # )  # [batch, intermediate_size, seq_len]
        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        discrete_time_step = self.dt_proj(
            time_step
        )  # [batch, seq_len, intermediate_size]
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(
            1, 2
        )  # [batch, intermediate_size, seq_len]
        # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        A = -torch.exp(self.A_log.float())  # [intermediate_size, ssm_state_size]
        discrete_A = torch.exp(
            A[None, :, None, :] * discrete_time_step[:, :, :, None]
        )  # [batch, intermediate_size, seq_len, ssm_state_size]
        discrete_B = (
            discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
        )  # [batch, intermediade_size, seq_len, ssm_state_size]
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()
        ### tree scan
        weight = rearrange(discrete_A, "b d l n -> b (d n) l").contiguous()
        feature_in = rearrange(deltaB_u, "b d l n -> b (d n) l").contiguous()
        feature_in = torch.flip(feature_in, dims=[-1]).contiguous()
        weight = torch.roll(torch.flip(weight, dims=[-1]), 1, -1).contiguous()

        mst = _MST.apply
        bfs = _BFS.apply
        refine = _Refine.apply

        ### hand-build tree
        tree_ = []
        for i in range(seq_len - 1):
            tree_.append([i, i + 1])
        tree_ = torch.tensor(tree_, dtype=torch.int32).to(device)
        tree = tree_.repeat(batch_size, 1, 1)
        sorted_index1, sorted_parent1, sorted_child1 = bfs(tree, 4)

        ### build tree by feature
        try:
            context_len = min(context_len)
        except:
            context_len = context_len
        with torch.no_grad():

            def generate_pairs(L, prompt_len):
                pairs = []
                for i in range(0, L - prompt_len):
                    pairs.append([i, i + 1])
                for i in range(L - prompt_len, L - 3):
                    pairs.append([i, i + 1])
                    pairs.append([i, i + 2])
                    pairs.append([i, i + 3])
                pairs.append([L - 3, L - 2])
                pairs.append([L - 3, L - 1])
                pairs.append([L - 2, L - 1])
                return pairs

            if context_len > 2:
                pairs = torch.tensor(
                    generate_pairs(seq_len, context_len),
                    dtype=torch.int32,
                    device=feature_in.device,
                )
                data1 = torch.index_select(feature_in, 2, pairs[:, 0])
                data2 = torch.index_select(feature_in, 2, pairs[:, 1])
                # import pdb;pdb.set_trace()
                tree_weight = cosine_distance(data1, data2)

                tree = mst(pairs.repeat(batch_size, 1, 1), tree_weight, seq_len)
                sorted_index2, sorted_parent2, sorted_child2 = bfs(tree, context_len)
            else:
                sorted_index2, sorted_parent2, sorted_child2 = (
                    sorted_index1,
                    sorted_parent1,
                    sorted_child1,
                )

            # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        feature_out1 = refine(
            feature_in, weight, sorted_index1, sorted_parent1, sorted_child1
        )
        # import pdb;pdb.set_trace()
        edge_weight = batch_index_opr(weight, sorted_index2)
        feature_out2 = refine(
            feature_in, edge_weight, sorted_index2, sorted_parent2, sorted_child2
        )
        feature_out = (
            feature_out2 * 0.3 + feature_out1
        )  # 0.3 is scaling factor (hyperparameter)

        feature_out = rearrange(
            torch.flip(feature_out.to(dtype), dims=[-1]),
            "b (d n) l -> b l d n",
            b=batch_size,
            n=discrete_A.shape[-1],
        ).contiguous()
        scan_output_ = (
            (feature_out @ C.unsqueeze(-1)).squeeze(-1).transpose(-1, -2)
        )  # (B, L, D, N) @ (B, L, N, 1) -> (B, L, D, 1)

        # [batch, seq_len, intermediade_size]
        scan_output = scan_output_ + (hidden_states * self.D[None, :, None])
        scan_output = scan_output * self.act(gate)
        # 4. Final linear projection
        contextualized_states = self.out_proj(
            scan_output.transpose(1, 2)
        )  # [batch, seq_len, hidden_size]
        return contextualized_states

    def step(self, hidden_states, conv_state, ssm_state):
        print("stepping...")
        dtype = hidden_states.dtype
        assert (
            hidden_states.shape[1] == 1
        ), "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(
                torch.roll(conv_state, shifts=-1, dims=-1)
            )  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state,
                x,
                dt,
                A,
                B,
                C,
                self.D,
                z=z,
                dt_bias=self.dt_proj.bias,
                dt_softplus=True,
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_conv,
            device=device,
            dtype=conv_dtype,
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def _get_states_from_cache(
        self, inference_params, batch_size, initialize_states=False
    ):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (
                conv_state,
                ssm_state,
            )
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[
                self.layer_idx
            ]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
        ids_restore=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(
            hidden_states, inference_params=inference_params, ids_restore=ids_restore
        )
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )
