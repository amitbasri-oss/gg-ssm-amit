from torch.autograd import Function
from torch.autograd.function import once_differentiable
from tree_scan_lan import _C
import torch
import torch.nn as nn
from einops import rearrange, repeat
import math


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


def tree_scanning_algorithm(self, input_states, context_len):
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype
    device = input_states.device
    # 1. Gated MLP's linear projection
    projected_states = self.in_proj(input_states).transpose(
        1, 2
    )  # [batch, 2 * intermediate_size, seq_len]
    hidden_states, gate = projected_states.chunk(2, dim=1)

    hidden_states = self.act(
        self.conv1d(hidden_states)[..., :seq_len]
    )  # [batch, intermediate_size, seq_len]
    # 3. State Space Model sequence transformation
    # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(
        ssm_parameters,
        [self.dt_rank, self.d_state, self.d_state],
        dim=-1,
    )
    discrete_time_step = self.dt_proj(time_step)  # [batch, seq_len, intermediate_size]
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


class GraphSSM(nn.Module):
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
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
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
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )

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

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
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

    def forward(self, input_states, context_len):
        return tree_scanning_algorithm(self, input_states, context_len)


if __name__ == "__main__":
    # Example hyperparameters
    d_model = 16
    seq_len = 12
    batch_size = 2
    context_len = 4  # Or pass in a list, e.g., [4, 4] for each sample

    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model)

    # Instantiate the GraphSSM layer
    model = GraphSSM(d_model=d_model)

    # Forward pass
    output = model(x, context_len)

    print("Input shape:", x.shape)  # (B, L, d_model)
    print("Output shape:", output.shape)  # (B, L, d_model)
    # Now 'output' contains the contextualized representation
