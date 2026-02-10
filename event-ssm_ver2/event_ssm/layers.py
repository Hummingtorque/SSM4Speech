import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional
import math


class EventPooling(nn.Module):
    """
    Subsampling layer for event sequences operating on batched inputs.
    """
    def __init__(self, stride: int = 1, mode: str = "last", eps: float = 1e-6):
        super().__init__()
        self.stride = stride
        self.mode = mode
        self.eps = eps

    def forward(self, x: torch.Tensor, integration_timesteps: torch.Tensor):
        """
        x: [B, L, H]
        integration_timesteps: [B, L]
        returns: [B, L/stride, H], [B, L/stride]
        """
        if self.stride == 1:
            return x, integration_timesteps

        B, L, H = x.shape
        new_len = (L // self.stride) * self.stride
        x = x[:, new_len - new_len:new_len, :]  # ensure positive slice
        t = integration_timesteps[:, new_len - new_len:new_len]

        steps = new_len // self.stride
        x = x.reshape(B, steps, self.stride, H)
        t = t.reshape(B, steps, self.stride)

        if self.mode == 'last':
            x = x[:, :, -1, :]
            t = t.sum(dim=2)
        elif self.mode == 'avgpool':
            x = x.mean(dim=2)
            t = t.sum(dim=2)
        elif self.mode == 'timepool':
            w = t + self.eps
            x = (x * w.unsqueeze(-1)).sum(dim=2)
            denom = w.sum(dim=2).clamp_min(self.eps)
            x = x / denom.unsqueeze(-1)
            t = denom
        else:
            raise NotImplementedError(f"Pooling mode: {self.mode} not implemented")

        return x, t


class TorchS5(nn.Module):
    """
    Minimal PyTorch S5-like block (diagonal state matrix) with on-the-fly discretization.
    Notes:
    - Implements a naive scan over time (loop on L). This is correct but not the
      most efficient; good for functional parity checks and small sanity runs.
    - Supports discretization='zoh', 'dirac', 'async' (async falls back to zoh-like gamma).
    """
    def __init__(
        self,
        H_in: int,
        H_out: int,
        P: int,
        block_size: int,
        discretization: str = "zoh",
        dt_min: float = 0.004,
        dt_max: float = 0.1,
        step_rescale: float = 1.0,
        stride: int = 1,
        pooling_mode: str = "last",
    ):
        super().__init__()
        self.H_in = H_in
        self.H_out = H_out
        self.P = P
        self.block_size = block_size
        self.discretization = discretization
        self.step_rescale = step_rescale
        self.stride = stride
        self.pool = EventPooling(stride=stride, mode=pooling_mode) if stride > 1 else None

        # Parameters
        # Initialize Lambda from a real-valued HiPPO (DPLR-inspired) spectrum for stability
        with torch.no_grad():
            # HiPPO-LegS construction
            idx = torch.arange(P, dtype=torch.float32)
            Pv = torch.sqrt(1.0 + 2.0 * idx)                       # (sqrt weights)
            A = torch.tril(Pv[:, None] * Pv[None, :]) - torch.diag(idx)  # Lower-triangular - diag
            A = -A                                                 # LegS is negative stable
            # Rank-1 term to form S (normalized variant)
            Pr = torch.sqrt(idx + 0.5)
            S = A + Pr[:, None] * Pr[None, :]
            # Symmetrize to ensure real eigvals for initialization
            S_sym = 0.5 * (S + S.T)
            # Compute eigenvalues on CPU to avoid GPU solver overhead
            eigvals = torch.linalg.eigvalsh(S_sym.cpu()).to(torch.float32)
            # Use strictly negative initialization (scale slightly)
            lambda_init = -torch.clamp(eigvals.abs(), min=1e-3)
        # Store a raw parameter and reparameterize to strictly negative Lambda during forward
        self.Lambda_raw = nn.Parameter(-torch.log(torch.expm1((-lambda_init).clamp_min(1e-6))))
        # Log step in [log(dt_min), log(dt_max)]
        log_step_min = torch.log(torch.tensor(dt_min))
        log_step_max = torch.log(torch.tensor(dt_max))
        log_step_init = (log_step_min + log_step_max) / 2.0
        self.log_step = nn.Parameter(log_step_init * torch.ones(P))  # [P]

        # B: (P, H_in), C: (H_out, P)
        self.B = nn.Parameter(torch.empty(P, H_in))
        self.C = nn.Parameter(torch.empty(H_out, P))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.C, a=math.sqrt(5))

        # D: passthrough term
        if H_in == H_out:
            self.D = nn.Parameter(torch.zeros(H_in))
        else:
            self.D = nn.Parameter(torch.empty(H_out, H_in))
            nn.init.kaiming_uniform_(self.D, a=math.sqrt(5))

    @staticmethod
    def _gamma_from_discretization(discretization: str, Lambda: torch.Tensor, Delta: torch.Tensor):
        # Lambda: [B, P] or [P], Delta: [B, P]
        eps = 1e-6
        if discretization in ["dirac"]:
            gamma = torch.ones_like(Delta)
        else:
            # zoh / async fall back to zoh-like formula
            # Clamp exponent to avoid overflow even if Lambda * Delta is large in magnitude
            A = torch.exp((Lambda * Delta).clamp(max=20.0))
            denom = Lambda
            # Broadcast denom to [B,P] if needed
            if denom.dim() == 1 and Delta.dim() == 2:
                denom = denom.unsqueeze(0).expand_as(Delta)
            gamma = torch.where(denom.abs() > eps, (A - 1.0) / denom, Delta)
        return gamma

    def forward(self, u: torch.Tensor, timesteps: torch.Tensor):
        """
        u: [B, L, H_in]
        timesteps: [B, L]
        returns y: [B, L, H_out]
        """
        Bsz, L, Hin = u.shape
        device = u.device
        P = self.P

        # Project inputs: Bu: [B, L, P]
        Bu = torch.einsum('ph,blh->blp', self.B, u)

        # Per-state step scaling and Lambda
        step_vec = self.step_rescale * torch.exp(self.log_step)  # [P]
        # Strictly negative Lambda to ensure stable discretization
        min_neg = 1e-3
        Lambda = -(F.softplus(self.Lambda_raw) + min_neg).to(device)  # [P]

        # Broadcast to [B, L, P]
        Delta = step_vec.view(1, 1, P) * timesteps.unsqueeze(-1)  # [B, L, P]
        # Optional safety clamp on Delta to avoid extremely large steps
        Delta = Delta.clamp_min(0.0)
        # Compute A_bar with clamped exponent to prevent overflow
        exp_arg = (Lambda.view(1, 1, P) * Delta).clamp(max=20.0)  # [B, L, P]
        A_bar = torch.exp(exp_arg)                                 # [B, L, P]
        gamma = self._gamma_from_discretization(self.discretization, Lambda.view(1, P), Delta)  # [B, L, P]

        # b_k = gamma ⊙ (B u_k)
        b_seq = gamma * Bu                                        # [B, L, P]

        # Vectorized diagonal recurrence:
        # x_k = (∏_{j=1..k} A_j) * sum_{i=1..k} b_i / (∏_{j=1..i} A_j)
        eps = 1e-9
        A_cum = torch.cumprod(A_bar, dim=1)                       # [B, L, P]
        z = b_seq / (A_cum + eps)                                 # [B, L, P]
        s = torch.cumsum(z, dim=1)                                # [B, L, P]
        x_seq = A_cum * s                                         # [B, L, P]  (x0 = 0)

        # Readout y = C x + D u
        ys = torch.einsum('hp,blp->blh', self.C, x_seq)           # [B, L, H_out]

        # Add D * u passthrough
        if self.H_in == self.H_out:
            ys = ys + (self.D.to(device) * u)
        else:
            ys = ys + torch.einsum('ho,blo->blh', self.D.to(device), u)

        # Optional stride pooling
        if self.pool is not None:
            ys, _ = self.pool(ys, timesteps)
        return ys


# class SequenceStage(nn.Module):
#     """
#     Defines a block of EventSSM layers with the same hidden size and event-resolution

#     :param ssm: the SSM to be used (i.e. S5 ssm)
#     :param d_model_in: this is the feature size of the layer inputs and outputs
#                         we usually refer to this size as H
#     :param d_model_out: this is the feature size of the layer outputs
#     :param d_ssm: the size of the state space model
#     :param ssm_block_size: the block size of the state space model
#     :param layers_per_stage: the number of S5 layers to stack
#     :param dropout: dropout rate
#     :param prenorm: whether to use layernorm before the module or after it
#     :param batchnorm: If True, use batchnorm instead of layernorm
#     :param bn_momentum: momentum for batchnorm
#     :param step_rescale: rescale the integration timesteps by this factor
#     :param pooling_stride: stride for pooling
#     :param pooling_mode: pooling mode (last, avgpool, timepool)
#     :param state_expansion_factor: factor to expand the state space model
#     """
#     ssm: nn.Module
#     discretization: str
#     d_model_in: int
#     d_model_out: int
#     d_ssm: int
#     ssm_block_size: int
#     layers_per_stage: int
#     dropout: float = 0.0
#     prenorm: bool = False
#     batchnorm: bool = False
#     bn_momentum: float = 0.9
#     step_rescale: float = 1.0
#     pooling_stride: int = 1
#     pooling_mode: str = "last"
#     state_expansion_factor: int = 1

#     @nn.compact
#     def __call__(self, x, integration_timesteps, train: bool):
#         """
#         Compute the LxH output of the stacked encoder given an Lxd_input input sequence.

#         :param x: input sequence (L, d_input)
#         :param integration_timesteps: the integration timesteps for the SSM
#         :param train: If True, applies dropout and batch norm from batch statistics
#         :return: output sequence (L, d_model), integration_timesteps
#         """
#         EventSSMLayer = partial(
#             SequenceLayer,
#             ssm=self.ssm,
#             discretization=self.discretization,
#             dropout=self.dropout,
#             d_ssm=self.d_ssm,
#             block_size=self.ssm_block_size,
#             prenorm=self.prenorm,
#             batchnorm=self.batchnorm,
#             bn_momentum=self.bn_momentum,
#             step_rescale=self.step_rescale,
#         )

#         # first layer with pooling
#         x, integration_timesteps = EventSSMLayer(
#             d_model_in=self.d_model_in,
#             d_model_out=self.d_model_out,
#             pooling_stride=self.pooling_stride,
#             pooling_mode=self.pooling_mode
#         )(x, integration_timesteps, train=train)

#         # further layers without pooling
#         for l in range(self.layers_per_stage - 1):
#             x, integration_timesteps = EventSSMLayer(
#                 d_model_in=self.d_model_out,
#                 d_model_out=self.d_model_out,
#                 pooling_stride=1
#             )(x, integration_timesteps, train=train)

#         return x, integration_timesteps

class SequenceStage(nn.Module):
    def __init__(
        self,
        ssm: nn.Module,
        discretization: str,
        d_model_in: int,
        d_model_out: int,
        d_ssm: int,
        ssm_block_size: int,
        layers_per_stage: int,
        dropout: float = 0.0,
        prenorm: bool = False,
        batchnorm: bool = False,
        bn_momentum: float = 0.9,
        step_rescale: float = 1.0,
        pooling_stride: int = 1,
        pooling_mode: str = "last",
        state_expansion_factor: int = 1,
        stage_index: int = 0,
        stage0_layer0_lambda: float = None,
        other_layers_lambda: float = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for l in range(layers_per_stage):
            stride = pooling_stride if l == 0 else 1
            layer = SequenceLayer(
                ssm=ssm,
                discretization=discretization,
                dropout=dropout,
                d_model_in=d_model_in if l == 0 else d_model_out,
                d_model_out=d_model_out,
                d_ssm=d_ssm,
                block_size=ssm_block_size,
                prenorm=prenorm,
                batchnorm=batchnorm,
                bn_momentum=bn_momentum,
                step_rescale=step_rescale,
                pooling_stride=stride,
                pooling_mode=pooling_mode,
            )
            self.layers.append(layer)

    def forward(self, x: torch.Tensor, integration_timesteps: torch.Tensor, train: bool):
        for layer in self.layers:
            x, integration_timesteps = layer(x, integration_timesteps, train=train)
        return x, integration_timesteps



# class SequenceLayer(nn.Module):
#     """
#     Defines a single event-ssm layer, with S5 SSM, nonlinearity,
#     dropout, batch/layer norm, etc.

#     :param ssm: the SSM to be used (i.e. S5 ssm)
#     :param discretization: the discretization method to use (zoh, dirac, async)
#     :param dropout: dropout rate
#     :param d_model_in: the input feature size
#     :param d_model_out: the output feature size
#     :param d_ssm: the size of the state space model
#     :param block_size: the block size of the state space model
#     :param prenorm: whether to use layernorm before the module or after it
#     :param batchnorm: If True, use batchnorm instead of layernorm
#     :param bn_momentum: momentum for batchnorm
#     :param step_rescale: rescale the integration timesteps by this factor
#     :param pooling_stride: stride for pooling
#     :param pooling_mode: pooling mode (last, avgpool, timepool)
#     """
#     ssm: nn.Module
#     discretization: str
#     dropout: float
#     d_model_in: int
#     d_model_out: int
#     d_ssm: int
#     block_size: int
#     prenorm: bool = False
#     batchnorm: bool = False
#     bn_momentum: float = 0.90
#     step_rescale: float = 1.0
#     pooling_stride: int = 1
#     pooling_mode: str = "last"

#     @nn.compact
#     def __call__(self, x, integration_timesteps, train: bool):
#         """
#         Compute a layer step

#         :param x: input sequence (L, d_model_in)
#         :param integration_timesteps: the integration timesteps for the SSM
#         :param train: If True, applies dropout and batch norm from batch statistics
#         :return: output sequence (L, d_model_out), integration_timesteps
#         """
#         skip = x

#         if self.prenorm:
#             norm = nn.BatchNorm(momentum=self.bn_momentum, axis_name='batch') if self.batchnorm else nn.LayerNorm()
#             x = norm(x, use_running_average=not train) if self.batchnorm else norm(x)

#         # apply state space model
#         x = self.ssm(
#             H_in=self.d_model_in, H_out=self.d_model_out, P=self.d_ssm, block_size=self.block_size,
#             step_rescale=self.step_rescale, discretization=self.discretization,
#             stride=self.pooling_stride, pooling_mode=self.pooling_mode
#         )(x, integration_timesteps)

#         # non-linear activation function
#         x1 = nn.Dropout(self.dropout, broadcast_dims=[0], deterministic=not train)(nn.gelu(x))
#         x1 = nn.Dense(self.d_model_out)(x1)
#         x = x * nn.sigmoid(x1)
#         x = nn.Dropout(self.dropout, broadcast_dims=[0], deterministic=not train)(x)

#         if self.pooling_stride > 1:
#             pool = EventPooling(stride=self.pooling_stride, mode=self.pooling_mode)
#             skip, integration_timesteps = pool(skip, integration_timesteps)

#         if self.d_model_in != self.d_model_out:
#             skip = nn.Dense(self.d_model_out)(skip)

#         x = skip + x

#         if not self.prenorm:
#             norm = nn.BatchNorm(momentum=self.bn_momentum, axis_name='batch') if self.batchnorm else nn.LayerNorm()
#             x = norm(x, use_running_average=not train) if self.batchnorm else norm(x)

#         return x, integration_timesteps
# -------------------------------
# 修改后的 SequenceLayer 模块
# -------------------------------
class SequenceLayer(nn.Module):
    def __init__(
        self,
        ssm: nn.Module,
        discretization: str,
        dropout: float,
        d_model_in: int,
        d_model_out: int,
        d_ssm: int,
        block_size: int,
        prenorm: bool = False,
        batchnorm: bool = False,
        bn_momentum: float = 0.90,
        step_rescale: float = 1.0,
        pooling_stride: int = 1,
        pooling_mode: str = "last",
    ):
        super().__init__()
        self.prenorm = prenorm
        self.batchnorm = batchnorm
        self.dropout_p = dropout
        self.pooling_stride = pooling_stride
        self.pooling_mode = pooling_mode

        # Optional SSM block factory (callable) -> build layer-local SSM
        self.ssm_block: Optional[nn.Module] = None
        if callable(ssm):
            # Build a concrete SSM module for this layer
            self.ssm_block = ssm(
                d_model_in, d_model_out, d_ssm, block_size,
                discretization, step_rescale, pooling_stride, pooling_mode
            )

        # Simple temporal block: linear projection + gated linear unit style (fallback / post-SSM nonlinearity)
        self.proj = nn.Linear(d_model_in, d_model_out)
        self.gate = nn.Linear(d_model_out, d_model_out)
        self.dropout = nn.Dropout(dropout)

        # Normalization layers
        if batchnorm:
            self.norm_in = nn.BatchNorm1d(d_model_out, momentum=bn_momentum)
            self.norm_out = nn.BatchNorm1d(d_model_out, momentum=bn_momentum)
        else:
            self.norm_in = nn.LayerNorm(d_model_out)
            self.norm_out = nn.LayerNorm(d_model_out)

        # Residual projection if needed
        self.skip_proj = nn.Linear(d_model_in, d_model_out) if d_model_in != d_model_out else None

        # Optional temporal pooling
        self.pool = EventPooling(stride=pooling_stride, mode=pooling_mode) if pooling_stride > 1 else None

    def _apply_norm(self, norm: nn.Module, x: torch.Tensor, train: bool) -> torch.Tensor:
        # BN1d expects [B, C, L]
        if isinstance(norm, nn.BatchNorm1d):
            x_perm = x.transpose(1, 2)
            x_norm = norm(x_perm)
            return x_norm.transpose(1, 2)
        return norm(x)

    def forward(self, x: torch.Tensor, integration_timesteps: torch.Tensor, train: bool):
        # x: [B, L, H_in]
        skip = x
        if self.ssm_block is not None:
            y = self.ssm_block(x, integration_timesteps)  # [B, L, H_out]
        else:
            y = self.proj(x)  # [B, L, H_out]

        if self.prenorm:
            y = self._apply_norm(self.norm_in, y, train)

        g = self.gate(self.dropout(F.gelu(y)) if train else F.gelu(y))
        y = y * torch.sigmoid(g)
        y = self.dropout(y) if train else y

        if self.pool is not None:
            if self.ssm_block is not None:
                # TorchS5 already applies pooling internally when stride > 1.
                # Pool only the skip branch to align sequence lengths.
                skip, integration_timesteps = self.pool(skip, integration_timesteps)
            else:
                # Fallback (no SSM): pool both branches consistently.
                orig_timesteps = integration_timesteps
                skip, integration_timesteps = self.pool(skip, integration_timesteps)
                y, _ = self.pool(y, orig_timesteps)

        if self.skip_proj is not None:
            skip = self.skip_proj(skip)

        y = y + skip

        if not self.prenorm:
            y = self._apply_norm(self.norm_out, y, train)

        return y, integration_timesteps
