import os
import math
import hydra
from omegaconf import OmegaConf as om
from omegaconf import DictConfig, open_dict
import torch
import torch.nn as nn

from event_ssm.dataloading import Datasets
from event_ssm.seq_model import ClassificationModel
from event_ssm.layers import TorchS5


def setup_evaluation(cfg: DictConfig):
    assert cfg.checkpoint, "No checkpoint provided. Use checkpoint=<path>."

    # load task specific data
    create_dataset_fn = Datasets[cfg.task.name]
    print("[*] Loading dataset...")
    train_loader, val_loader, test_loader, data = create_dataset_fn(
        cache_dir=cfg.data_dir,
        seed=cfg.seed,
        world_size=1,
        **cfg.training
    )

    print("[*] Creating model...")
    # Provide safe defaults for optional SSM fields that may be absent in configs
    ssm_cfg = cfg.model.ssm
    step_rescale = float(ssm_cfg.get('step_rescale', 1.0))
    pooling_every_n_layers = int(ssm_cfg.get('pooling_every_n_layers', 1))
    ssm_init = cfg.model.ssm_init
    def s5_factory(d_model_in, d_model_out, d_ssm, block_size, discretization, step_rescale_layer, stride, pooling_mode):
        return TorchS5(
            H_in=d_model_in,
            H_out=d_model_out,
            P=d_ssm,
            block_size=block_size,
            discretization=discretization,
            dt_min=float(ssm_init.dt_min),
            dt_max=float(ssm_init.dt_max),
            step_rescale=float(step_rescale_layer),
            stride=int(stride),
            pooling_mode=str(pooling_mode),
        )
    audio_cfg = cfg.model.get("audio_encoder", {})
    input_is_mel = bool(audio_cfg.get("use", False))
    audio_encoder_type = str(audio_cfg.get("type", "conv1d"))
    audio_in_channels = int(audio_cfg.get("in_channels", 1))
    audio_freq_stride = int(audio_cfg.get("freq_stride", 1))
    conv_kernel = int(audio_cfg.get("kernel_size", 3))
    conv_stride = int(audio_cfg.get("stride", 1))
    mel_bins = int(getattr(cfg.training, "mel_bins", 0))
    num_embeddings = 0 if input_is_mel else data.num_embeddings
    model = ClassificationModel(
        ssm=s5_factory,
        discretization=cfg.model.ssm.discretization,
        num_classes=data.n_classes,
        d_model=cfg.model.ssm.d_model,
        d_ssm=cfg.model.ssm.d_ssm,
        ssm_block_size=cfg.model.ssm.ssm_block_size,
        num_stages=cfg.model.ssm.num_stages,
        num_layers_per_stage=cfg.model.ssm.num_layers_per_stage,
        num_embeddings=num_embeddings,
        input_is_mel=input_is_mel,
        mel_bins=mel_bins,
        audio_encoder_type=audio_encoder_type,
        audio_in_channels=audio_in_channels,
        audio_freq_stride=audio_freq_stride,
        conv_kernel=conv_kernel,
        conv_stride=conv_stride,
        dropout=cfg.model.ssm.dropout,
        classification_mode=cfg.model.ssm.classification_mode,
        prenorm=cfg.model.ssm.prenorm,
        batchnorm=cfg.model.ssm.batchnorm,
        bn_momentum=cfg.model.ssm.bn_momentum,
        step_rescale=step_rescale,
        pooling_stride=cfg.model.ssm.pooling_stride,
        pooling_every_n_layers=pooling_every_n_layers,
        pooling_mode=cfg.model.ssm.pooling_mode,
        state_expansion_factor=cfg.model.ssm.state_expansion_factor,
    )
    return model, train_loader, val_loader, test_loader, data


@hydra.main(version_base=None, config_path='configs', config_name='base')
def main(config: DictConfig):
    print(om.to_yaml(config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, _, test_loader, data = setup_evaluation(config)
    model.to(device)

    ckpt_path = config.checkpoint
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, 'model.pt')
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['state_dict'])

    criterion = nn.CrossEntropyLoss()

    def numpy_batch_to_tensors(batch):
        inputs, targets, timesteps, lengths = batch
        x = torch.from_numpy(inputs).to(device)
        y = torch.from_numpy(targets).argmax(dim=-1).to(device)
        t = torch.from_numpy(timesteps).to(device)
        l = torch.from_numpy(lengths).to(device)
        return x, y, t, l

    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x, y, t, l = numpy_batch_to_tensors(batch)
            logits = model(x, t, l, train=False)
            loss_sum += criterion(logits, y).item()
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.numel()
    acc = 100.0 * correct / max(1, total)
    loss = loss_sum / max(1, len(test_loader))
    print(f"[*] Test accuracy: {acc:.3f}% | loss {loss:.4f}")


if __name__ == '__main__':
    main()
