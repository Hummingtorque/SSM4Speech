import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import SequenceStage


class InceptionConv2dFrontend(nn.Module):
    """
    Inception-like Conv2D frontend for mel-spectrograms.
    Input: [B, 1, mel_bins, L]
    Output: [B, L, d_model]
    """
    def __init__(self, d_model: int, mel_bins: int, in_channels: int = 1, time_stride: int = 1, freq_stride: int = 1):
        super().__init__()
        branch_ch = max(1, d_model // 4)
        self.in_channels = int(in_channels)
        self.time_stride = int(time_stride)
        self.freq_stride = int(freq_stride)
        self.freq_bins = (int(mel_bins) - 1) // max(1, self.freq_stride) + 1
        # Four branches: 1x1, 3x3, 5x5, 3x3 dilated
        self.b1 = nn.Sequential(
            nn.Conv2d(self.in_channels, branch_ch, kernel_size=1, stride=(self.freq_stride, self.time_stride), padding=0),
            nn.BatchNorm2d(branch_ch),
            nn.SiLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(self.in_channels, branch_ch, kernel_size=3, stride=(self.freq_stride, self.time_stride), padding=1),
            nn.BatchNorm2d(branch_ch),
            nn.SiLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(self.in_channels, branch_ch, kernel_size=5, stride=(self.freq_stride, self.time_stride), padding=2),
            nn.BatchNorm2d(branch_ch),
            nn.SiLU(inplace=True),
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(self.in_channels, branch_ch, kernel_size=3, stride=(self.freq_stride, self.time_stride), padding=2, dilation=2),
            nn.BatchNorm2d(branch_ch),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(branch_ch * 4, d_model, kernel_size=1, padding=0),
            nn.BatchNorm2d(d_model),
            nn.SiLU(inplace=True),
        )
        # Project frequency dimension into channels (BC-ResNet style)
        self.freq_proj = nn.Conv1d(d_model * self.freq_bins, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, mel_bins, L]
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        y = self.proj(y)  # [B, d_model, F, L']
        B, C, F, T = y.shape
        # Flatten frequency into channels then 1x1 Conv to d_model
        y = y.permute(0, 1, 3, 2).reshape(B, C * F, T)  # [B, C*F, T]
        y = self.freq_proj(y)  # [B, d_model, T]
        y = y.transpose(1, 2)  # [B, T, d_model]
        return y


class StackedEncoderModel(nn.Module):
    """
    PyTorch implementation of the stacked encoder previously built with Flax/JAX.
    """
    def __init__(
        self,
        ssm: nn.Module,
        discretization: str,
        d_model: int,
        d_ssm: int,
        ssm_block_size: int,
        num_stages: int,
        num_layers_per_stage: int,
        num_embeddings: int = 0,
        input_is_mel: bool = False,
        mel_bins: int = 0,
        audio_encoder_type: str = "conv1d",
        audio_in_channels: int = 1,
        audio_freq_stride: int = 1,
        conv_kernel: int = 3,
        conv_stride: int = 1,
        dropout: float = 0.0,
        prenorm: bool = False,
        batchnorm: bool = False,
        bn_momentum: float = 0.9,
        step_rescale: float = 1.0,
        pooling_stride: int = 1,
        pooling_every_n_layers: int = 1,
        pooling_mode: str = "last",
        state_expansion_factor: int = 1,
        stage0_layer0_lambda: float = -0.45,
        other_layers_lambda: float = -0.45,
    ):
        super().__init__()
        self.input_is_mel = input_is_mel
        self.audio_encoder_type = str(audio_encoder_type)
        self.conv_stride = int(conv_stride)
        if self.input_is_mel:
            assert mel_bins > 0, "mel_bins must be > 0 when using mel input"
            if self.audio_encoder_type == "inception2d":
                self.inception_encoder = InceptionConv2dFrontend(
                    d_model=d_model,
                    mel_bins=mel_bins,
                    in_channels=audio_in_channels,
                    time_stride=self.conv_stride,
                    freq_stride=audio_freq_stride,
                )
            else:
                # Conv1d encoder over mel bins; output becomes token embeddings
                pad = int(conv_kernel) // 2
                self.conv_encoder = nn.Conv1d(
                    in_channels=int(mel_bins),
                    out_channels=int(d_model),
                    kernel_size=int(conv_kernel),
                    stride=self.conv_stride,
                    padding=pad,
                )
        else:
            assert num_embeddings > 0, "num_embeddings must be > 0 for token input"
            # Add +1 for padding index 0; tokens will be shifted by +1
            self.encoder = nn.Embedding(num_embeddings + 1, d_model, padding_idx=0)
        self.pooling_stride = pooling_stride
        self.num_stages = num_stages

        stages = []
        d_model_in = d_model
        d_model_out = d_model
        current_d_ssm = d_ssm
        total_downsampling = max(1, self.conv_stride)
        for stage_idx in range(num_stages):
            total_downsampling *= pooling_stride
            stages.append(
                SequenceStage(
                    ssm=ssm,
                    discretization=discretization,
                    d_model_in=d_model_in,
                    d_model_out=d_model_out,
                    d_ssm=current_d_ssm,
                    ssm_block_size=ssm_block_size,
                    layers_per_stage=num_layers_per_stage,
                    dropout=dropout,
                    prenorm=prenorm,
                    batchnorm=batchnorm,
                    bn_momentum=bn_momentum,
                    step_rescale=step_rescale,
                    pooling_stride=pooling_stride,
                    pooling_mode=pooling_mode,
                    state_expansion_factor=state_expansion_factor,
                    stage_index=stage_idx,
                    stage0_layer0_lambda=stage0_layer0_lambda if stage_idx == 0 else None,
                    other_layers_lambda=other_layers_lambda
                )
            )
            current_d_ssm = state_expansion_factor * current_d_ssm
            d_model_out = state_expansion_factor * d_model_in
            if stage_idx > 0:
                d_model_in = state_expansion_factor * d_model_in
        self.stages = nn.ModuleList(stages)
        self.total_downsampling = total_downsampling

    def forward(self, x: torch.Tensor, integration_timesteps: torch.Tensor, train: bool):
        if self.input_is_mel:
            if self.audio_encoder_type == "inception2d":
                # x: [B, L, mel_bins] -> [B, 1, mel_bins, L]
                x = x.transpose(1, 2).unsqueeze(1)
                x = self.inception_encoder(x)  # [B, L, d_model]
            else:
                # x: [B, L, mel_bins] -> Conv1d expects [B, mel_bins, L]
                x = x.transpose(1, 2)
                x = self.conv_encoder(x)
                x = F.silu(x)
                x = x.transpose(1, 2)  # [B, L, d_model]
            if self.conv_stride > 1:
                integration_timesteps = integration_timesteps[:, ::self.conv_stride]
        else:
            # x: [B, L] tokens in [-1..num_embeddings-1]; shift by +1 to map padding -1 -> 0
            x = x.long() + 1
            x = self.encoder(x)  # [B, L, d_model]
        for stage in self.stages:
            x, integration_timesteps = stage(x, integration_timesteps, train=train)
        return x, integration_timesteps


def masked_meanpool(x: torch.Tensor, lengths: torch.Tensor):
    """
    Helper function to perform mean pooling across the sequence length
    when sequences have variable lengths. We only want to pool across
    the prepadded sequence length.

    :param x: input sequence (L, d_model)
    :param lengths: the original length of the sequence before padding
    :return: mean pooled output sequence (d_model)
    """
    B, L, H = x.shape
    device = x.device
    mask = torch.arange(L, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    masked = x * mask.unsqueeze(-1)
    denom = lengths.clamp_min(1).unsqueeze(-1).to(x.dtype)
    return masked.sum(dim=1) / denom


def timepool(x: torch.Tensor, integration_timesteps: torch.Tensor, eps: float = 1e-6):
    """
    Helper function to perform weighted mean across the sequence length.
    Means are weighted with the integration time steps

    :param x: input sequence (L, d_model)
    :param integration_timesteps: the integration timesteps for the SSM
    :return: time pooled output sequence (d_model)
    """
    weights = integration_timesteps.clamp_min(0.0).unsqueeze(-1) + eps
    integral = (x * weights).sum(dim=1)
    T = weights.sum(dim=1).clamp_min(eps)
    return integral / T


def masked_timepool(x: torch.Tensor, lengths: torch.Tensor, integration_timesteps: torch.Tensor, eps: float = 1e-6):
    """
    Helper function to perform weighted mean across the sequence length
    when sequences have variable lengths. We only want to pool across
    the prepadded sequence length. Means are weighted with the integration time steps

    :param x: input sequence (L, d_model)
    :param lengths: the original length of the sequence before padding
    :param integration_timesteps: the integration timesteps for the SSM
    :param eps: small value to avoid division by zero
    :return: time pooled output sequence (d_model)
    """
    B, L, H = x.shape
    device = x.device
    mask = (torch.arange(L, device=device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
    weights = (integration_timesteps.clamp_min(0.0).unsqueeze(-1) + eps)
    integral = (x * mask * weights).sum(dim=1)
    T = (weights * mask).sum(dim=1).clamp_min(eps)
    return integral / T


def masked_attn_pool(x: torch.Tensor, lengths: torch.Tensor, attn_proj: nn.Module):
    """
    Attention pooling across time with padding mask.

    :param x: [B, L, H]
    :param lengths: [B]
    :param attn_proj: nn.Linear(H, 1)
    :return: pooled [B, H]
    """
    B, L, _ = x.shape
    device = x.device
    mask = torch.arange(L, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    scores = attn_proj(x).squeeze(-1)  # [B, L]
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, L, 1]
    return (x * weights).sum(dim=1)


class ClassificationModel(nn.Module):
    """
    EventSSM classification sequence model in PyTorch.
    """
    def __init__(
        self,
        ssm: nn.Module,
        discretization: str,
        num_classes: int,
        d_model: int,
        d_ssm: int,
        ssm_block_size: int,
        num_stages: int,
        num_layers_per_stage: int,
        num_embeddings: int = 0,
        input_is_mel: bool = False,
        mel_bins: int = 0,
        audio_encoder_type: str = "conv1d",
        audio_in_channels: int = 1,
        audio_freq_stride: int = 1,
        conv_kernel: int = 3,
        conv_stride: int = 1,
        dropout: float = 0.2,
        classification_mode: str = "pool",
        prenorm: bool = False,
        batchnorm: bool = False,
        bn_momentum: float = 0.9,
        step_rescale: float = 1.0,
        pooling_stride: int = 1,
        pooling_every_n_layers: int = 1,
        pooling_mode: str = "last",
        state_expansion_factor: int = 1,
    ):
        super().__init__()
        self.classification_mode = classification_mode
        self.encoder = StackedEncoderModel(
            ssm=ssm,
            discretization=discretization,
            d_model=d_model,
            d_ssm=d_ssm,
            ssm_block_size=ssm_block_size,
            num_stages=num_stages,
            num_layers_per_stage=num_layers_per_stage,
            num_embeddings=num_embeddings,
            input_is_mel=input_is_mel,
            mel_bins=mel_bins,
            audio_encoder_type=audio_encoder_type,
            audio_in_channels=audio_in_channels,
            audio_freq_stride=audio_freq_stride,
            conv_kernel=conv_kernel,
            conv_stride=conv_stride,
            dropout=dropout,
            prenorm=prenorm,
            batchnorm=batchnorm,
            bn_momentum=bn_momentum,
            step_rescale=step_rescale,
            pooling_stride=pooling_stride,
            pooling_every_n_layers=pooling_every_n_layers,
            pooling_mode=pooling_mode,
            state_expansion_factor=state_expansion_factor
        )
        self.decoder = nn.Linear(d_model * (state_expansion_factor ** (num_stages - 1)), num_classes)
        self.attn_pool = nn.Linear(d_model * (state_expansion_factor ** (num_stages - 1)), 1)

    def forward(self, x: torch.Tensor, integration_timesteps: torch.Tensor, length: torch.Tensor, train: bool = True):
        # Adjust length for downsampling
        length = length // self.encoder.total_downsampling

        x, integration_timesteps = self.encoder(x, integration_timesteps, train=train)

        if self.classification_mode in ["pool"]:
            x = masked_meanpool(x, length)
        elif self.classification_mode in ["timepool"]:
            x = masked_timepool(x, length, integration_timesteps)
        elif self.classification_mode in ["attnpool"]:
            x = masked_attn_pool(x, length, self.attn_pool)
        elif self.classification_mode in ["last"]:
            # Last valid state
            idx = (length - 1).clamp_min(0)
            x = x[torch.arange(x.size(0), device=x.device), idx]
        else:
            raise NotImplementedError("Mode must be in ['pool', 'timepool', 'attnpool', 'last']")

        x = self.decoder(x)
        return x


# For compatibility with previous import name
BatchClassificationModel = ClassificationModel
