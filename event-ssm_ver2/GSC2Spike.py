import numpy as np
import librosa
from typing import Optional
try:
    import scipy.signal as sp_signal
except Exception:
    sp_signal = None


def wav_to_spike_events(
    wav_path: str,
    mel_bins: int = 128,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 160,   # ~10 ms at 16 kHz
    win_length: int = 400,   # ~25 ms at 16 kHz
    # Audio frontend improvements
    pre_emphasis: float = 0.97,
    use_log_mel: bool = True,
    top_db: float = 80.0,
    # Event sampling controls
    event_rate_scale: float = 1.5,
    prob_power: float = 0.7,
    min_prob: float = 0.005,
) -> dict:
    """
    Convert a GSC .wav file into an event stream over mel bins.
    - Events are returned as a dict with keys:
      't': int64 timestamps in microseconds (non-decreasing),
      'x': int32 mel-bin indices in [0, mel_bins-1].
    - Uses Bernoulli sampling per (mel_bin, frame) from normalized mel energy.
      The default pipeline applies pre-emphasis and log-mel compression for better robustness.
    """
    # Load mono audio at target sample rate
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    if y.size == 0:
        # Fallback: create two dummy events to keep downstream happy
        return {
            "t": np.array([0, 1000], dtype=np.int64),
            "x": np.array([0, 0], dtype=np.int32),
        }

    # Pre-emphasis for high-frequency boost (optional)
    if pre_emphasis and sp_signal is not None and pre_emphasis > 0.0:
        y = sp_signal.lfilter([1.0, -pre_emphasis], [1.0], y)

    # Mel power spectrogram: shape [mel_bins, n_frames]
    mel_power = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=mel_bins,
        power=2.0,
    )

    # Dynamic range compression: log-mel in [0,1]
    if use_log_mel:
        # dB scaled relative to per-file max with a cap on dynamic range
        mel_db = librosa.power_to_db(mel_power, ref=np.max, top_db=float(top_db))
        # mel_db in [-top_db, 0], map to [0,1]
        mel_norm = ((mel_db + float(top_db)) / float(top_db)).astype(np.float32)
    else:
        # Linear normalization to [0,1]
        max_val = float(np.max(mel_power)) if mel_power.size > 0 else 0.0
        if max_val <= 0.0 or not np.isfinite(max_val):
            mel_norm = np.zeros_like(mel_power, dtype=np.float32)
        else:
            mel_norm = (mel_power / (max_val + 1e-8)).astype(np.float32)

    n_mels, n_frames = mel_norm.shape

    # Frame timestamps in microseconds (use frame centers)
    frame_times_sec = librosa.frames_to_time(
        np.arange(n_frames, dtype=np.int64), sr=sr, hop_length=hop_length, n_fft=n_fft
    )
    frame_times_us = (frame_times_sec * 1e6).astype(np.int64)

    rng = np.random.default_rng()
    events_t: list[int] = []
    events_x: list[int] = []

    # Bernoulli sampling per (mel_bin, frame) with probability transformed:
    # p = clip(min_prob + event_rate_scale * (mel_norm ** prob_power), 0, 1)
    # Keep time strictly non-decreasing by adding tiny intra-frame offsets
    for f in range(n_frames):
        p_col = mel_norm[:, f]
        p_col = np.clip(min_prob + event_rate_scale * np.power(p_col, prob_power), 0.0, 1.0)
        rand_col = rng.random(size=n_mels, dtype=np.float32)
        active_bins = np.nonzero(rand_col < p_col)[0]
        if active_bins.size == 0:
            continue
        base_t = frame_times_us[f]
        # Small intra-frame offset in microseconds to keep monotonicity
        for j, b in enumerate(active_bins):
            events_t.append(base_t + j)
            events_x.append(int(b))

    # Ensure at least 2 events to avoid empty/degenerate sequences downstream
    if len(events_t) < 2:
        # Pick up to two strongest (mel, frame) locations deterministically
        flat = mel_norm.reshape(-1)
        if flat.size == 0 or not np.isfinite(flat).any():
            return {
                "t": np.array([0, 1000], dtype=np.int64),
                "x": np.array([0, 0], dtype=np.int32),
            }
        top_idx = np.argpartition(-flat, kth=min(1, flat.size - 1))[:2]
        top_idx = top_idx[np.argsort(-flat[top_idx])]
        for rank, idx in enumerate(top_idx):
            b = int(idx // n_frames)
            f = int(idx % n_frames)
            t = int(frame_times_us[f] + rank)
            events_t.append(t)
            events_x.append(int(b))

    # Convert to numpy arrays with expected dtypes
    t_arr = np.asarray(events_t, dtype=np.int64)
    x_arr = np.asarray(events_x, dtype=np.int32)

    # Final safety: clamp indices into valid range
    if mel_bins > 0:
        x_arr = np.clip(x_arr, 0, mel_bins - 1).astype(np.int32, copy=False)

    # Sort by time just in case (should already be non-decreasing)
    order = np.argsort(t_arr, kind="stable")
    t_arr = t_arr[order]
    x_arr = x_arr[order]

    return {"t": t_arr, "x": x_arr}


