import os
from functools import partial
from pathlib import Path
import hydra
from omegaconf import OmegaConf as om
from omegaconf import DictConfig, open_dict
import math
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim

from event_ssm.dataloading import Datasets
from event_ssm.seq_model import ClassificationModel
from event_ssm.layers import TorchS5


def setup_training(cfg: DictConfig):
    # For safety, default to single-process training
    num_devices = 1

    # load task specific data
    create_dataset_fn = Datasets[cfg.task.name]

    print("[*] Loading dataset...")
    train_loader, val_loader, test_loader, data = create_dataset_fn(
        cache_dir=cfg.data_dir,
        seed=cfg.seed,
        world_size=num_devices,
        **cfg.training
    )

    with open_dict(cfg):
        cfg.optimizer.total_steps = cfg.training.num_epochs * len(train_loader) // max(1, cfg.optimizer.accumulation_steps)
        cfg.optimizer.warmup_steps = cfg.optimizer.warmup_epochs * len(train_loader) // max(1, cfg.optimizer.accumulation_steps)
        cfg.optimizer.ssm_lr = cfg.optimizer.ssm_base_lr * cfg.training.per_device_batch_size * num_devices * max(1, cfg.optimizer.accumulation_steps)

    print("[*] Creating model...")
    # Provide safe defaults for optional SSM fields that may be absent in configs
    ssm_cfg = cfg.model.ssm
    step_rescale = float(ssm_cfg.get('step_rescale', 1.0))
    pooling_every_n_layers = int(ssm_cfg.get('pooling_every_n_layers', 1))
    # Build S5 factory (callable) capturing ssm_init hyperparameters
    ssm_init = cfg.model.ssm_init
    input_gate = bool(ssm_cfg.get("input_gate", False))
    input_gate_rank = int(ssm_cfg.get("input_gate_rank", 0))
    input_gate_mode = str(ssm_cfg.get("input_gate_mode", "sigmoid"))
    input_gate_energy_scale_init = float(ssm_cfg.get("input_gate_energy_scale_init", 0.0))
    input_gate_bias_init = float(ssm_cfg.get("input_gate_bias_init", 0.0))
    input_gate_min = float(ssm_cfg.get("input_gate_min", 0.0))
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
            input_gate=input_gate,
            input_gate_rank=input_gate_rank,
            input_gate_mode=input_gate_mode,
            input_gate_energy_scale_init=input_gate_energy_scale_init,
            input_gate_bias_init=input_gate_bias_init,
            input_gate_min=input_gate_min,
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
    os.makedirs(os.path.join(config.logging.log_dir, 'checkpoints'), exist_ok=True)
    with open(os.path.join(config.logging.log_dir, 'config.yaml'), 'w') as f:
        om.save(config, f)
    wandb = None
    wandb_run = None
    try:
        import wandb as _wandb
        wandb = _wandb
        wandb_run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "event-ssm_ver2"),
            config=om.to_container(config, resolve=True),
            dir=config.logging.log_dir,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("val_accuracy", summary="max")
        wandb.define_metric("accuracy", summary="max")
    except Exception as exc:
        # Keep training robust even when W&B is unavailable/misconfigured.
        print(f"[warn] W&B disabled: {exc}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, train_loader, val_loader, test_loader, data = setup_training(config)
    model.to(device)

    # Optimizer and loss
    lr = float(config.optimizer.ssm_lr)
    # Separate parameter groups: SSM vs regular (to mirror JAX multi_transform)
    ssm_weight_decay = float(getattr(config.optimizer, 'ssm_weight_decay', 0.0))
    regular_weight_decay = float(config.optimizer.weight_decay)
    lr_factor = float(config.optimizer.lr_factor)

    # Collect SSM parameters by module type
    from event_ssm.layers import TorchS5
    ssm_param_ids = set()
    for module in model.modules():
        if isinstance(module, TorchS5):
            for p in module.parameters(recurse=False):
                ssm_param_ids.add(id(p))
            # Also include child parameters (B, C, D, Lambda, log_step live here)
            for p in module.parameters(recurse=True):
                ssm_param_ids.add(id(p))

    ssm_params = []
    regular_params = []
    for p in model.parameters():
        (ssm_params if id(p) in ssm_param_ids else regular_params).append(p)

    param_groups = [
        {'params': ssm_params, 'lr': lr, 'weight_decay': ssm_weight_decay},
        {'params': regular_params, 'lr': lr * lr_factor, 'weight_decay': regular_weight_decay},
    ]
    # Remember initial_lrs to apply the same schedule ratio to each group
    for g in param_groups:
        g['initial_lr'] = g['lr']
    optimizer = optim.AdamW(param_groups)
    # Label smoothing (optional)
    label_smoothing = float(getattr(config.optimizer, 'label_smoothing', 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing if label_smoothing > 0 else 0.0)
    accumulation_steps = max(1, int(config.optimizer.accumulation_steps))
    # Gradient clipping (optional)
    grad_clip_norm = float(getattr(config.optimizer, 'grad_clip_norm', 0.0))
    # Warmup + cosine schedule
    total_steps = int(getattr(config.optimizer, 'total_steps', 0))
    warmup_steps = int(getattr(config.optimizer, 'warmup_steps', 0))
    schedule_kind = str(getattr(config.optimizer, 'schedule', 'cosine'))
    # EMA config
    use_ema = bool(getattr(config.optimizer, 'use_ema', True))
    ema_decay = float(getattr(config.optimizer, 'ema_decay', 0.999))
    eval_with_ema = bool(getattr(config.optimizer, 'eval_ema', True))

    # Prepare EMA parameter dictionary
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    ema_params = {n: p.detach().clone() for n, p in named_params} if use_ema else {}

    # Optional gate-direction regularization:
    # encourage mean_gate(keyword) > mean_gate(background) + margin
    gate_reg_weight = float(getattr(config.optimizer, "gate_reg_weight", 0.0))
    gate_reg_margin = float(getattr(config.optimizer, "gate_reg_margin", 0.05))
    gate_reg_bg_batch_size = int(getattr(config.optimizer, "gate_reg_bg_batch_size", 8))
    gate_reg_interval = int(getattr(config.optimizer, "gate_reg_interval", 4))
    gate_reg_bg_dir = str(getattr(config.optimizer, "gate_reg_bg_dir", ""))
    gate_reg_enabled = gate_reg_weight > 0.0 and gate_reg_bg_batch_size > 0 and gate_reg_interval > 0
    gate_reg_bg_mels = []
    gate_reg_rng = np.random.default_rng(int(getattr(config, "seed", 1234)))
    gate_reg_dt_ms = float(getattr(config.training, "hop_length", 160)) / float(getattr(config.training, "sr", 16000)) * 1000.0
    gate_reg_pad_unit = int(getattr(config.training, "pad_unit", 8192))
    gate_reg_mel_bins = int(getattr(config.training, "mel_bins", 64))
    gate_reg_sr = int(getattr(config.training, "sr", 16000))
    gate_reg_n_fft = int(getattr(config.training, "n_fft", 1024))
    gate_reg_hop = int(getattr(config.training, "hop_length", 160))
    gate_reg_win = int(getattr(config.training, "win_length", 400))
    gate_reg_top_db = float(getattr(config.training, "top_db", 80.0))
    gate_reg_use_log_mel = bool(getattr(config.training, "use_log_mel", True))

    def _load_bg_mel_cache():
        nonlocal gate_reg_bg_mels, gate_reg_enabled
        if not gate_reg_enabled:
            return
        bg_dir = Path(gate_reg_bg_dir) if gate_reg_bg_dir else (Path(str(config.training.root)) / "_background_noise_")
        if not bg_dir.exists():
            print(f"[warn] Gate reg background dir not found: {bg_dir}. Disabling gate regularization.")
            gate_reg_enabled = False
            return
        wavs = sorted(bg_dir.glob("*.wav"))
        if len(wavs) == 0:
            print(f"[warn] No background wavs in {bg_dir}. Disabling gate regularization.")
            gate_reg_enabled = False
            return
        for wavp in wavs:
            y, _ = librosa.load(str(wavp), sr=gate_reg_sr, mono=True)
            mel_power = librosa.feature.melspectrogram(
                y=y,
                sr=gate_reg_sr,
                n_fft=gate_reg_n_fft,
                hop_length=gate_reg_hop,
                win_length=gate_reg_win,
                n_mels=gate_reg_mel_bins,
                power=2.0,
            )
            if gate_reg_use_log_mel:
                mel_db = librosa.power_to_db(mel_power, ref=np.max, top_db=float(gate_reg_top_db))
                mel = ((mel_db + float(gate_reg_top_db)) / float(gate_reg_top_db)).astype(np.float32)
            else:
                max_val = float(np.max(mel_power)) if mel_power.size > 0 else 0.0
                if max_val <= 0.0 or not np.isfinite(max_val):
                    mel = np.zeros_like(mel_power, dtype=np.float32)
                else:
                    mel = (mel_power / (max_val + 1e-8)).astype(np.float32)
            mel_t = mel.T
            if mel_t.shape[0] > 0:
                gate_reg_bg_mels.append(mel_t)
        if len(gate_reg_bg_mels) == 0:
            print(f"[warn] Background mel cache empty from {bg_dir}. Disabling gate regularization.")
            gate_reg_enabled = False
            return
        print(f"[*] Gate regularization enabled. Background clips loaded: {len(gate_reg_bg_mels)}")

    def _collect_gate_per_sample() -> torch.Tensor | None:
        vals = []
        for module in model.modules():
            if isinstance(module, TorchS5) and getattr(module, "input_gate", False):
                g = getattr(module, "_last_gate_per_sample", None)
                if g is not None:
                    vals.append(g)
        if len(vals) == 0:
            return None
        # Stack over layers -> [num_layers, B], then average over layers.
        return torch.stack(vals, dim=0).mean(dim=0)

    def _sample_background_batch(lengths: torch.Tensor):
        # lengths: [B]
        lengths_np = lengths.detach().cpu().numpy().astype(np.int32)
        Bbg = min(gate_reg_bg_batch_size, int(lengths_np.shape[0]))
        if Bbg <= 0:
            return None
        picked_idx = gate_reg_rng.choice(lengths_np.shape[0], size=Bbg, replace=(lengths_np.shape[0] < Bbg))
        picked_lengths = np.maximum(1, lengths_np[picked_idx])
        max_len = int(np.max(picked_lengths))
        pad_len = (max_len // gate_reg_pad_unit) * gate_reg_pad_unit + gate_reg_pad_unit
        x = np.zeros((Bbg, pad_len, gate_reg_mel_bins), dtype=np.float32)
        t = np.zeros((Bbg, pad_len), dtype=np.float32)
        l = picked_lengths.astype(np.int32)
        for i, li in enumerate(picked_lengths):
            mel = gate_reg_bg_mels[int(gate_reg_rng.integers(0, len(gate_reg_bg_mels)))]
            if mel.shape[0] >= li:
                st = int(gate_reg_rng.integers(0, mel.shape[0] - li + 1))
                seg = mel[st:st + li]
            else:
                seg = np.zeros((li, gate_reg_mel_bins), dtype=np.float32)
                seg[:mel.shape[0], :] = mel
            x[i, :li, :] = seg
            t[i, :li] = gate_reg_dt_ms
        return (
            torch.from_numpy(x).to(device),
            torch.from_numpy(t).to(device),
            torch.from_numpy(l).to(device),
        )

    _load_bg_mel_cache()

    def get_scheduled_lr(step_idx: int) -> float:
        if total_steps <= 0:
            return lr
        if step_idx < warmup_steps and warmup_steps > 0:
            return lr * float(step_idx + 1) / float(max(1, warmup_steps))
        if schedule_kind in ['cosine', 'cosine_decay']:
            progress = 0.0
            if total_steps > warmup_steps:
                progress = float(step_idx - warmup_steps) / float(max(1, total_steps - warmup_steps))
                progress = min(max(progress, 0.0), 1.0)
            return 0.5 * lr * (1.0 + math.cos(math.pi * progress))
        # constant
        return lr

    def numpy_batch_to_tensors(batch):
        inputs, targets, timesteps, lengths = batch
        x = torch.from_numpy(inputs).to(device)
        y = torch.from_numpy(targets).argmax(dim=-1).to(device)  # one-hot -> indices
        t = torch.from_numpy(timesteps).to(device)
        l = torch.from_numpy(lengths).to(device)
        return x, y, t, l

    def ema_update():
        if not use_ema:
            return
        with torch.no_grad():
            for n, p in named_params:
                if n in ema_params:
                    ema_params[n].mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)

    def _swap_in_params(param_dict: dict):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in param_dict:
                    p.copy_(param_dict[n].to(p.device))

    def _eval_with_params(run_eval_fn, use_ema_params: bool):
        if not use_ema_params or not use_ema:
            return run_eval_fn()
        # Backup current params
        backup_params = {n: p.detach().clone() for n, p in model.named_parameters()}
        try:
            _swap_in_params(ema_params)
            return run_eval_fn()
        finally:
            _swap_in_params(backup_params)

    best_val_acc = -1.0
    global_step = 0
    for epoch in range(1, int(config.training.num_epochs) + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        optimizer.zero_grad(set_to_none=True)

        # Stability monitor: track max Lambda (after reparam) if TorchS5 is used
        def _ssm_max_lambda_abs():
            max_val = None
            for module in model.modules():
                if isinstance(module, TorchS5):
                    # Recreate negative Lambda as in forward: -(softplus(raw)+eps)
                    lamb = -(torch.nn.functional.softplus(module.Lambda_raw) + 1e-3)
                    cur = lamb.abs().max().detach().item()
                    max_val = cur if max_val is None else max(max_val, cur)
            return max_val

        for step, batch in enumerate(train_loader, 1):
            x, y, t, l = numpy_batch_to_tensors(batch)
            logits = model(x, t, l, train=True)
            kw_gate = _collect_gate_per_sample()
            # NaN/Inf guard on logits before loss
            if not torch.isfinite(logits).all():
                print(f"[warn] Non-finite logits encountered at epoch {epoch}, step {step}. Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue
            gate_reg_loss = torch.zeros((), device=device)
            if gate_reg_enabled and (step % gate_reg_interval == 0) and (kw_gate is not None):
                bg_batch = _sample_background_batch(l)
                if bg_batch is not None:
                    bg_x, bg_t, bg_l = bg_batch
                    _ = model(bg_x, bg_t, bg_l, train=True)
                    bg_gate = _collect_gate_per_sample()
                    if bg_gate is not None:
                        gate_reg_loss = torch.relu(gate_reg_margin - (kw_gate.mean() - bg_gate.mean()))
            loss = (criterion(logits, y) + gate_reg_weight * gate_reg_loss) / accumulation_steps
            # NaN/Inf guard on loss
            if not torch.isfinite(loss):
                print(f"[warn] Non-finite loss encountered at epoch {epoch}, step {step}. Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()

            running_loss += loss.item() * accumulation_steps
            running_correct += (logits.argmax(dim=-1) == y).sum().item()
            running_total += y.numel()

            if step % accumulation_steps == 0:
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                # update LR before optimizer step
                scheduled_abs_lr = get_scheduled_lr(global_step)
                # Apply schedule as a ratio relative to ssm base lr so groups keep their multipliers
                ratio = scheduled_abs_lr / max(lr, 1e-12)
                for g in optimizer.param_groups:
                    base = g.get('initial_lr', lr)
                    g['lr'] = base * ratio
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                # EMA update after optimizer step
                ema_update()

            if step % int(config.logging.interval) == 0:
                acc = 100.0 * running_correct / max(1, running_total)
                # Print current learning rate of first param group for reference
                current_lr_pg0 = optimizer.param_groups[0]['lr']
                print(f"| epoch {epoch} | step {step}/{len(train_loader)} | loss {running_loss/step:.4f} | acc {acc:.2f}% | lr {current_lr_pg0:.2e}")

        # Validation
        model.eval()
        max_lambda_abs = _ssm_max_lambda_abs()
        def run_val():
            val_correct = 0
            val_total = 0
            val_loss_sum = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x, y, t, l = numpy_batch_to_tensors(batch)
                    logits = model(x, t, l, train=False)
                    if not torch.isfinite(logits).all():
                        continue
                    val_loss_sum += criterion(logits, y).item()
                    val_correct += (logits.argmax(dim=-1) == y).sum().item()
                    val_total += y.numel()
            return 100.0 * val_correct / max(1, val_total), val_loss_sum / max(1, len(val_loader))
        val_acc, val_loss = _eval_with_params(run_val, eval_with_ema)
        msg = f"| Eval epoch {epoch} | val_loss {val_loss:.4f} | val_acc {val_acc:.2f}%"
        if max_lambda_abs is not None:
            msg = f"{msg} | max|Lambda| {max_lambda_abs:.4f}"
        print(msg)
        if wandb_run is not None:
            train_acc = 100.0 * running_correct / max(1, running_total)
            train_loss = running_loss / max(1, len(train_loader))
            wandb.log({
                "epoch": epoch,
                "val_accuracy": float(val_acc),
                "accuracy": float(val_acc),
                "train_accuracy": float(train_acc),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "best_accuracy": float(max(best_val_acc, val_acc)),
                "lr": float(optimizer.param_groups[0]["lr"]),
            })

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(config.logging.log_dir, 'checkpoints', 'model.pt')
            # Save EMA weights if available, else current weights
            if use_ema and eval_with_ema:
                # Build a state_dict with EMA tensors injected
                state = model.state_dict()
                for k in list(state.keys()):
                    # Map only parameter keys that exist in ema_params
                    if k in ema_params:
                        state[k] = ema_params[k].detach().to(state[k].device)
                torch.save({'state_dict': state}, ckpt_path)
            else:
                torch.save({'state_dict': model.state_dict()}, ckpt_path)

    # Test (optional)
    if test_loader is not None:
        model.eval()
        def run_test():
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for batch in test_loader:
                    x, y, t, l = numpy_batch_to_tensors(batch)
                    logits = model(x, t, l, train=False)
                    test_correct += (logits.argmax(dim=-1) == y).sum().item()
                    test_total += y.numel()
            return 100.0 * test_correct / max(1, test_total)
        test_acc = _eval_with_params(run_test, eval_with_ema)
        print(f"[*] Test accuracy: {test_acc:.2f}%")
        if wandb_run is not None:
            wandb.log({"test_accuracy": float(test_acc)})

    # Print Lambda values per SSM layer at end of training
    layer_idx = 0
    for module in model.modules():
        if isinstance(module, TorchS5):
            lamb = -(torch.nn.functional.softplus(module.Lambda_raw) + 1e-3)
            lamb_list = lamb.detach().cpu().numpy().tolist()
            print(f"[*] Lambda layer {layer_idx}: {lamb_list}")
            layer_idx += 1
    if wandb_run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()
