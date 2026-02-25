import torch
from pathlib import Path
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union
import tonic
from functools import partial
import numpy as np
import torchaudio
from event_ssm.transform import Identity, Roll, Rotate, Scale, DropEventChunk, Jitter1D, OneHotLabels, cut_mix_augmentation

DEFAULT_CACHE_DIR_ROOT = Path('./cache_dir/')

DataLoader = TypeVar('DataLoader')
InputType = [str, Optional[int], Optional[int]]


class Data:
    """
    Data class for storing dataset specific information
    """
    def __init__(
            self,
            n_classes: int,
            num_embeddings: int,
            train_size: int
    ):
        self.n_classes = n_classes
        self.num_embeddings = num_embeddings
        self.train_size = train_size


def event_stream_collate_fn(batch, resolution, pad_unit, cut_mix=0.0, no_time_information=False):
    """
    Collate function to turn event stream data into tokens ready for the JAX model

    :param batch: list of tuples of (events, target)
    :param resolution: resolution of the event stream
    :param pad_unit: padding unit for the tokens. All sequences will be padded to integer multiples of this unit.
                     This option results in JAX compiling multiple GPU kernels for different sequence lengths,
                     which might slow down compilation time, but improves throughput for the rest of the training process.
    :param cut_mix: probability of applying cut mix augmentation
    :param no_time_information: if True, the time information is ignored and all events are treated as if they were
                                recorded sampled at uniform time intervals.
                                This option is only used for ablation studies.
    """
    # x are inputs, y are targets, z are aux data
    x, y, *z = zip(*batch)
    assert len(z) == 0
    batch_size_one = len(x) == 1

    # apply cut mix augmentation
    if np.random.rand() < cut_mix:
        x, y = cut_mix_augmentation(x, y)

    # set labels to numpy array
    y = np.stack(y)

    # integration time steps are the difference between two consecutive time stamps
    if no_time_information:
        timesteps = [np.ones_like(e['t'][:-1], dtype=np.float32) for e in x]
    else:
        # Compute deltas, ensure finite and non-negative to avoid NaNs downstream
        timesteps = []
        for e in x:
            dt = np.diff(e['t']).astype(np.float32)
            dt = np.nan_to_num(dt, nan=0.0, posinf=0.0, neginf=0.0)
            dt = np.maximum(dt, 0.0)
            timesteps.append(dt)

    # NOTE: since timesteps are deltas, their length is L - 1, and we have to remove the last token in the following

    # process tokens for single input dim (e.g. audio)
    if len(resolution) == 1:
        tokens = []
        max_idx = int(np.prod(resolution)) - 1
        for e in x:
            idx = e['x'][:-1].astype(np.float32)
            idx = np.nan_to_num(idx, nan=-1.0, posinf=-1.0, neginf=-1.0)
            idx = np.clip(idx, -1.0, float(max_idx)).astype(np.int32)
            tokens.append(idx)
    elif len(resolution) == 2:
        tokens = []
        max_idx = int(np.prod(resolution)) * 2 - 1  # include polarity plane
        width = int(resolution[0])
        height = int(resolution[1])
        for e in x:
            ex = np.nan_to_num(e['x'][:-1].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            ey = np.nan_to_num(e['y'][:-1].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            ep = np.nan_to_num(e['p'][:-1].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            ex = np.clip(ex, 0.0, float(width - 1)).astype(np.int32)
            ey = np.clip(ey, 0.0, float(height - 1)).astype(np.int32)
            ep = (ep > 0).astype(np.int32)
            idx = (ex + width * ey) + (int(np.prod(resolution)) * ep)
            idx = np.clip(idx, -1, max_idx).astype(np.int32)
            tokens.append(idx)
    else:
        raise ValueError('resolution must contain 1 or 2 elements')

    # get padding lengths
    lengths = np.array([len(e) for e in timesteps], dtype=np.int32)
    pad_length = (lengths.max() // pad_unit) * pad_unit + pad_unit

    # pad tokens with -1, which results in a zero vector with embedding look-ups
    tokens = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=-1) for e in tokens])
    timesteps = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=0.0).astype(np.float32) for e in timesteps])

    # timesteps are in micro seconds... transform to milliseconds
    timesteps = timesteps / 1000

    if batch_size_one:
        lengths = lengths[None, ...]

    return tokens, y, timesteps, lengths


def event_stream_dataloader(
        train_data,
        val_data,
        test_data,
        batch_size,
        eval_batch_size,
        train_collate_fn,
        eval_collate_fn,
        rng,
        num_workers=0,
        shuffle_training=True
):
    """
    Create dataloaders for training, validation and testing

    :param train_data: training dataset
    :param val_data: validation dataset
    :param test_data: test dataset
    :param batch_size: batch size for training
    :param eval_batch_size: batch size for evaluation
    :param train_collate_fn: collate function for training
    :param eval_collate_fn: collate function for evaluation
    :param rng: random number generator
    :param num_workers: number of workers for data loading
    :param shuffle_training: whether to shuffle the training data

    :return: train_loader, val_loader, test_loader
    """
    def dataloader(dset, bsz, collate_fn, shuffle, drop_last):
        return torch.utils.data.DataLoader(
            dset,
            batch_size=bsz,
            drop_last=drop_last,
            collate_fn=collate_fn,
            shuffle=shuffle,
            generator=rng,
            num_workers=num_workers
        )
    train_loader = dataloader(train_data, batch_size, train_collate_fn, shuffle=shuffle_training, drop_last=True)
    val_loader = dataloader(val_data, eval_batch_size, eval_collate_fn, shuffle=False, drop_last=True)
    # 修改：测试集也设置 drop_last=True，确保 batch 大小能整除设备数
    test_loader = dataloader(test_data, eval_batch_size, eval_collate_fn, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader


def mel_spectrogram_collate_fn(
        batch,
        pad_unit,
        dt_ms: float,
        specaugment_prob: float = 0.0,
        freq_mask_param: int = 0,
        time_mask_param: int = 0,
        num_masks: int = 1,
        mixup_alpha: float = 0.0,
        mixup_prob: float = 0.0,
):
    """
    Collate function for mel-spectrogram sequences.
    Each item is (mel_frames [L, mel_bins], onehot).
    Returns padded mel [B, L, mel_bins], onehot [B, C], timesteps [B, L], lengths [B].
    """
    x, y, *z = zip(*batch)
    assert len(z) == 0
    y = np.stack(y)

    def _apply_specaugment(mel: np.ndarray) -> np.ndarray:
        if specaugment_prob <= 0.0 or (freq_mask_param <= 0 and time_mask_param <= 0):
            return mel
        if np.random.rand() > specaugment_prob:
            return mel
        out = mel.copy()
        L, C = out.shape
        for _ in range(max(1, int(num_masks))):
            if freq_mask_param > 0 and C > 1:
                f = np.random.randint(0, min(freq_mask_param, C))
                f0 = np.random.randint(0, max(1, C - f))
                out[:, f0:f0 + f] = 0.0
            if time_mask_param > 0 and L > 1:
                t = np.random.randint(0, min(time_mask_param, L))
                t0 = np.random.randint(0, max(1, L - t))
                out[t0:t0 + t, :] = 0.0
        return out

    # Apply SpecAugment per sample before padding
    x = [_apply_specaugment(m) for m in x]

    lengths = np.array([m.shape[0] for m in x], dtype=np.int32)
    pad_length = (lengths.max() // pad_unit) * pad_unit + pad_unit
    mel_bins = x[0].shape[1]

    mel = np.stack([
        np.pad(m, ((0, pad_length - m.shape[0]), (0, 0)), mode='constant', constant_values=0.0)
        for m in x
    ]).astype(np.float32)
    timesteps = np.stack([
        np.pad(np.full((m.shape[0],), dt_ms, dtype=np.float32), (0, pad_length - m.shape[0]), mode='constant', constant_values=0.0)
        for m in x
    ]).astype(np.float32)

    # Mixup on mel + labels (+timesteps) after padding
    if mixup_alpha > 0.0 and np.random.rand() < mixup_prob:
        lam = np.random.beta(mixup_alpha, mixup_alpha, size=mel.shape[0]).astype(np.float32)
        perm = np.random.permutation(mel.shape[0])
        lam_m = lam.reshape(-1, 1, 1)
        mel = lam_m * mel + (1.0 - lam_m) * mel[perm]
        y = lam.reshape(-1, 1) * y + (1.0 - lam.reshape(-1, 1)) * y[perm]
        # Blend timesteps; keep lengths as max to avoid over-masking
        timesteps = lam.reshape(-1, 1) * timesteps + (1.0 - lam.reshape(-1, 1)) * timesteps[perm]
        lengths = np.maximum(lengths, lengths[perm])

    return mel, y, timesteps, lengths


def create_events_shd_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 0,
        seed: int = 42,
        time_jitter: float = 100,
        spatial_jitter: float = 1.0,
        max_drop_chunk: float = 0.1,
        noise: int = 100,
        drop_event: float = 0.1,
        time_skew: float = 1.1,
        cut_mix: float = 0.5,
        pad_unit: int = 8192,
        validate_on_test: bool = False,
        no_time_information: bool = False,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    Creates a view of the Spiking Heidelberg Digits Classification Dataset.
    """
    print("[*] Generating Spiking Heidelberg Digits Classification Dataset")

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    sensor_size = (700, 1, 1)

    transforms = tonic.transforms.Compose([
        tonic.transforms.DropEvent(p=drop_event),
        DropEventChunk(p=0.3, max_drop_size=max_drop_chunk),
        Jitter1D(sensor_size=sensor_size, var=spatial_jitter),
        tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
        tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
        tonic.transforms.UniformNoise(sensor_size=sensor_size, n=(0, noise))
    ])
    target_transforms = OneHotLabels(num_classes=20)

    train_data = tonic.datasets.SHD(save_to=cache_dir, train=True, transform=transforms, target_transform=target_transforms)
    val_data = tonic.datasets.SHD(save_to=cache_dir, train=True, target_transform=target_transforms)
    test_data = tonic.datasets.SHD(save_to=cache_dir, train=False, target_transform=target_transforms)

    # create validation set
    if validate_on_test:
        print("[*] WARNING: Using test set for validation")
        val_data = tonic.datasets.SHD(save_to=cache_dir, train=False, target_transform=target_transforms)
    else:
        val_length = int(0.1 * len(train_data))
        indices = torch.randperm(len(train_data), generator=rng)
        train_data = torch.utils.data.Subset(train_data, indices[:-val_length])
        val_data = torch.utils.data.Subset(val_data, indices[-val_length:])

    collate_fn = partial(event_stream_collate_fn, resolution=(700,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix),
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )
    data = Data(
        n_classes=20, num_embeddings=700, train_size=len(train_data)
    )
    return train_loader, val_loader, test_loader, data


def create_events_ssc_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 0,
        seed: int = 42,
        time_jitter: float = 100,
        spatial_jitter: float = 1.0,
        max_drop_chunk: float = 0.1,
        noise: int = 100,
        drop_event: float = 0.1,
        time_skew: float = 1.1,
        cut_mix: float = 0.5,
        pad_unit: int = 8192,
        no_time_information: bool = False,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    Creates a view of the Spiking Speech Commands Classification Dataset.
    """
    print("[*] Generating Spiking Speech Commands Classification Dataset")

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    sensor_size = (700, 1, 1)

    transforms = tonic.transforms.Compose([
        tonic.transforms.DropEvent(p=drop_event),
        DropEventChunk(p=0.3, max_drop_size=max_drop_chunk),
        Jitter1D(sensor_size=sensor_size, var=spatial_jitter),
        tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
        tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
        tonic.transforms.UniformNoise(sensor_size=sensor_size, n=(0, noise))
    ])
    target_transforms = OneHotLabels(num_classes=35)

    train_data = tonic.datasets.SSC(save_to=cache_dir, split='train', transform=transforms, target_transform=target_transforms)
    val_data = tonic.datasets.SSC(save_to=cache_dir, split='valid', target_transform=target_transforms)
    test_data = tonic.datasets.SSC(save_to=cache_dir, split='test', target_transform=target_transforms)

    collate_fn = partial(event_stream_collate_fn, resolution=(700,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix),
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )

    data = Data(
        n_classes=35, num_embeddings=700, train_size=len(train_data)
    )
    return train_loader, val_loader, test_loader, data


def create_events_dvs_gesture_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 0,
        seed: int = 42,
        slice_events: int = 0,
        pad_unit: int = 2 ** 19,
        # Augmentation parameters
        time_jitter: float = 100,
        spatial_jitter: float = 1.0,
        noise: int = 100,
        drop_event: float = 0.1,
        time_skew: float = 1.1,
        cut_mix: float = 0.5,
        downsampling: int = 1,
        max_roll: int = 4,
        max_angle: float = 10,
        max_scale: float = 1.5,
        max_drop_chunk: float = 0.1,
        validate_on_test: bool = False,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    Creates a view of the DVS Gesture Classification Dataset.
    """
    print("[*] Generating DVS Gesture Classification Dataset")

    assert time_skew > 1, "time_skew must be greater than 1"

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    orig_sensor_size = (128, 128, 2)
    new_sensor_size = (128 // downsampling, 128 // downsampling, 2)
    train_transforms = [
        # Event transformations
        DropEventChunk(p=0.3, max_drop_size=max_drop_chunk),
        tonic.transforms.DropEvent(p=drop_event),
        tonic.transforms.UniformNoise(sensor_size=new_sensor_size, n=(0, noise)),
        # Time transformations
        tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
        tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
        # Spatial transformations
        tonic.transforms.SpatialJitter(sensor_size=orig_sensor_size, var_x=spatial_jitter, var_y=spatial_jitter, clip_outliers=True),
        tonic.transforms.Downsample(sensor_size=orig_sensor_size, target_size=new_sensor_size[:2]) if downsampling > 1 else Identity(),
        # Geometric transformations
        Roll(sensor_size=new_sensor_size, p=0.3, max_roll=max_roll),
        Rotate(sensor_size=new_sensor_size, p=0.3, max_angle=max_angle),
        Scale(sensor_size=new_sensor_size, p=0.3, max_scale=max_scale),
    ]

    train_transforms = tonic.transforms.Compose(train_transforms)
    test_transforms = tonic.transforms.Compose([
        tonic.transforms.Downsample(sensor_size=orig_sensor_size, target_size=new_sensor_size[:2]) if downsampling > 1 else Identity(),
    ])
    target_transforms = OneHotLabels(num_classes=11)

    TrainData = partial(tonic.datasets.DVSGesture, save_to=cache_dir, train=True)
    TestData = partial(tonic.datasets.DVSGesture, save_to=cache_dir, train=False)

    # create validation set
    if validate_on_test:
        print("[*] WARNING: Using test set for validation")
        val_data = TestData(transform=test_transforms, target_transform=target_transforms)
    else:
        # create train validation split
        val_data = TrainData(transform=test_transforms, target_transform=target_transforms)
        val_length = int(0.2 * len(val_data))
        indices = torch.randperm(len(val_data), generator=rng)
        val_data = torch.utils.data.Subset(val_data, indices[-val_length:])
    
    # if slice event count is given, train on slices of the training data
    if slice_events > 0:
        slicer = tonic.slicers.SliceByEventCount(event_count=slice_events, overlap=slice_events // 2, include_incomplete=True)
        train_subset = torch.utils.data.Subset(TrainData(), indices[:-val_length]) if not validate_on_test else TrainData()
        train_data = tonic.sliced_dataset.SlicedDataset(
            dataset=train_subset,
            slicer=slicer,
            transform=train_transforms,
            target_transform=target_transforms,
            metadata_path=None
        )
    else:
        train_data = torch.utils.data.Subset(
            TrainData(transform=train_transforms, target_transform=target_transforms),
            indices[:-val_length]
        ) if not validate_on_test else TrainData(transform=train_transforms)

    # Always evaluate on the full sequences
    test_data = TestData(transform=test_transforms, target_transform=target_transforms)

    # define collate functions
    train_collate_fn = partial(
            event_stream_collate_fn,
            resolution=new_sensor_size[:2],
            pad_unit=slice_events if (slice_events != 0 and slice_events < pad_unit) else pad_unit,
            cut_mix=cut_mix
        )
    eval_collate_fn = partial(
            event_stream_collate_fn,
            resolution=new_sensor_size[:2],
            pad_unit=pad_unit,
        )
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=train_collate_fn,
        eval_collate_fn=eval_collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )

    data = Data(
        n_classes=11, num_embeddings=np.prod(new_sensor_size), train_size=len(train_data)
    )
    return train_loader, val_loader, test_loader, data


def create_events_dvs_lip_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 0,
        seed: int = 42,
        slice_events: int = 0,
        pad_unit: int = 2 ** 19,
        time_jitter: float = 100,
        spatial_jitter: float = 1.0,
        noise: int = 100,
        drop_event: float = 0.1,
        time_skew: float = 1.1,
        cut_mix: float = 0.5,
        downsampling: int = 1,
        max_roll: int = 4,
        max_angle: float = 10,
        max_scale: float = 1.5,
        max_drop_chunk: float = 0.1,
        validate_on_test: bool = False,
        no_time_information: bool = False,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    Creates a view of the DVS-Lip Classification Dataset.
    DVSLip contains 100 word labels.
    """
    print("[*] Generating DVS-Lip Classification Dataset")

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    orig_sensor_size = (128, 128, 2)
    new_sensor_size = (128 // downsampling, 128 // downsampling, 2)

    # Define training transforms
    train_transforms = [
        # Event transformations
        DropEventChunk(p=0.3, max_drop_size=max_drop_chunk),
        tonic.transforms.DropEvent(p=drop_event),
        tonic.transforms.UniformNoise(sensor_size=new_sensor_size, n=(0, noise)),
        # Time transformations
        tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
        tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
        # Spatial transformations
        tonic.transforms.SpatialJitter(sensor_size=orig_sensor_size, var_x=spatial_jitter, var_y=spatial_jitter, clip_outliers=True),
        tonic.transforms.Downsample(sensor_size=orig_sensor_size, target_size=new_sensor_size[:2]) if downsampling > 1 else Identity(),
        # Geometric transformations
        Roll(sensor_size=new_sensor_size, p=0.3, max_roll=max_roll),
        Rotate(sensor_size=new_sensor_size, p=0.3, max_angle=max_angle),
        Scale(sensor_size=new_sensor_size, p=0.3, max_scale=max_scale),
    ]
    train_transforms = tonic.transforms.Compose(train_transforms)

    # For evaluation, we only apply downsampling (if needed)
    test_transforms = tonic.transforms.Compose([
        tonic.transforms.Downsample(sensor_size=orig_sensor_size, target_size=new_sensor_size[:2]) if downsampling > 1 else Identity(),
    ])

    target_transforms = OneHotLabels(num_classes=100)

    # Use tonic.datasets.DVSLip to load the dataset
    TrainData = partial(tonic.datasets.DVSLip, save_to=cache_dir, train=True)
    TestData = partial(tonic.datasets.DVSLip, save_to=cache_dir, train=False)

    # Generate validation set
    if validate_on_test:
        print("[*] WARNING: Using test set for validation")
        val_data = TestData(transform=test_transforms, target_transform=target_transforms)
        train_data = TrainData(transform=train_transforms, target_transform=target_transforms)
    else:
        full_train = TrainData(transform=test_transforms, target_transform=target_transforms)
        val_length = int(0.2 * len(full_train))
        indices = torch.randperm(len(full_train), generator=rng)
        val_data = torch.utils.data.Subset(full_train, indices[-val_length:])
        if slice_events > 0:
            slicer = tonic.slicers.SliceByEventCount(event_count=slice_events, overlap=slice_events // 2, include_incomplete=True)
            train_subset = torch.utils.data.Subset(
                TrainData(transform=train_transforms, target_transform=target_transforms), indices[:-val_length]
            )
            train_data = tonic.sliced_dataset.SlicedDataset(
                dataset=train_subset,
                slicer=slicer,
                transform=train_transforms,
                target_transform=target_transforms,
                metadata_path=None
            )
        else:
            train_data = torch.utils.data.Subset(
                TrainData(transform=train_transforms, target_transform=target_transforms), indices[:-val_length]
            )

    test_data = TestData(transform=test_transforms, target_transform=target_transforms)

    # Define collate functions
    train_collate_fn = partial(
        event_stream_collate_fn,
        resolution=new_sensor_size[:2],
        pad_unit=slice_events if (slice_events != 0 and slice_events < pad_unit) else pad_unit,
        cut_mix=cut_mix,
        no_time_information=no_time_information
    )
    eval_collate_fn = partial(
        event_stream_collate_fn,
        resolution=new_sensor_size[:2],
        pad_unit=pad_unit,
        no_time_information=no_time_information
    )
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=train_collate_fn,
        eval_collate_fn=eval_collate_fn,
        batch_size=per_device_batch_size * world_size,
        eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng,
        num_workers=num_workers,
        shuffle_training=True
    )

    data = Data(
        n_classes=100,
        num_embeddings=np.prod(new_sensor_size),
        train_size=len(train_data)
    )
    return train_loader, val_loader, test_loader, data


# -------------------------------
# Google Speech Commands (GSC) v0.02 via SNN spike encoding
# -------------------------------
def create_events_gsc_snn_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 0,
        seed: int = 42,
        pad_unit: int = 8192,
        cut_mix: float = 0.0,
        mel_bins: int = 64,
        root: Union[str, Path, None] = None,
        include_labels: Optional[List[str]] = None,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    Create a dataset for GSC v0.02 where each .wav is converted to spike events using snntorch (via GSC2Spike.py).
    Expects the GSC directory to contain validation_list.txt and testing_list.txt.
    """
    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    # Determine dataset root: prefer explicit 'root', else use cache_dir
    root_path = Path(root) if root is not None else Path(cache_dir)

    # Parse official splits
    val_list_path = root_path / "validation_list.txt"
    test_list_path = root_path / "testing_list.txt"
    assert val_list_path.exists() and test_list_path.exists(), "GSC split lists not found in root."
    val_list = set(p for p in val_list_path.read_text().splitlines() if p.strip())
    test_list = set(p for p in test_list_path.read_text().splitlines() if p.strip())

    # Label directories (exclude folders starting with underscore like _background_noise_)
    labels = sorted([d.name for d in root_path.iterdir() if d.is_dir() and not d.name.startswith("_")])
    if include_labels is not None:
        labels = [l for l in labels if l in include_labels]
    label_to_idx = {l: i for i, l in enumerate(labels)}

    def iter_files(split: str):
        for l in labels:
            for wavp in (root_path / l).glob("*.wav"):
                rel = f"{l}/{wavp.name}"
                if split == "train" and (rel not in val_list and rel not in test_list):
                    yield wavp, label_to_idx[l]
                elif split == "val" and rel in val_list:
                    yield wavp, label_to_idx[l]
                elif split == "test" and rel in test_list:
                    yield wavp, label_to_idx[l]

    class GSCSNNDataset(torch.utils.data.Dataset):
        def __init__(self, split: str):
            self.items = list(iter_files(split))

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx: int):
            # Import inside to avoid hard dependency unless this dataset is used
            from GSC2Spike import wav_to_spike_events
            wav_path, y_idx = self.items[idx]
            events = wav_to_spike_events(str(wav_path), mel_bins=mel_bins)
            # One-hot label
            onehot = np.eye(len(labels), dtype=np.float32)[y_idx]
            return events, onehot

    train_data = GSCSNNDataset("train")
    val_data = GSCSNNDataset("val")
    test_data = GSCSNNDataset("test")

    # Define collate using 1D resolution of mel_bins
    collate_fn = partial(
        event_stream_collate_fn,
        resolution=(mel_bins,),
        pad_unit=pad_unit,
        no_time_information=False
    )
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix),
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size,
        eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )

    data = Data(
        n_classes=len(labels),
        num_embeddings=mel_bins,
        train_size=len(train_data)
    )
    return train_loader, val_loader, test_loader, data


# -------------------------------
# Google Speech Commands (GSC) raw audio -> mel-spectrogram (no explicit spike conversion)
# -------------------------------
def create_audio_gsc_mel_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 0,
        seed: int = 42,
        pad_unit: int = 8192,
        mel_bins: int = 128,
        sr: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 160,
        win_length: int = 400,
        top_db: float = 80.0,
        use_log_mel: bool = True,
        root: Union[str, Path, None] = None,
        include_labels: Optional[List[str]] = None,
        specaugment_prob: float = 0.0,
        freq_mask_param: int = 0,
        time_mask_param: int = 0,
        num_masks: int = 1,
        mixup_alpha: float = 0.0,
        mixup_prob: float = 0.0,
        silence_ratio: float = 0.05,
        vad_threshold: float = 0.0,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    root_path = Path(root) if root is not None else Path(cache_dir)
    val_list_path = root_path / "validation_list.txt"
    test_list_path = root_path / "testing_list.txt"
    assert val_list_path.exists() and test_list_path.exists(), "GSC split lists not found in root."
    val_list = set(p for p in val_list_path.read_text().splitlines() if p.strip())
    test_list = set(p for p in test_list_path.read_text().splitlines() if p.strip())

    all_labels = sorted([d.name for d in root_path.iterdir() if d.is_dir() and not d.name.startswith("_")])
    include_labels = list(include_labels) if include_labels is not None else None
    want_silence = include_labels is not None and "silence" in include_labels
    want_unknown = include_labels is not None and "unknown" in include_labels

    if include_labels is not None:
        # Base known labels are the requested labels excluding special tokens
        labels = [l for l in include_labels if l not in ("silence", "unknown")]
    else:
        labels = list(all_labels)

    # Append special classes in a stable order if requested
    if want_unknown and "unknown" not in labels:
        labels.append("unknown")
    if want_silence and "silence" not in labels:
        labels.append("silence")

    label_to_idx = {l: i for i, l in enumerate(labels)}

    # Labels that should map to "unknown"
    unknown_sources = []
    if want_unknown:
        unknown_sources = [l for l in all_labels if l not in labels and l not in ("silence", "unknown")]

    def iter_files(split: str):
        # Known labels
        for l in labels:
            if l in ("silence", "unknown"):
                continue
            for wavp in (root_path / l).glob("*.wav"):
                rel = f"{l}/{wavp.name}"
                if split == "train" and (rel not in val_list and rel not in test_list):
                    yield wavp, label_to_idx[l]
                elif split == "val" and rel in val_list:
                    yield wavp, label_to_idx[l]
                elif split == "test" and rel in test_list:
                    yield wavp, label_to_idx[l]

        # Unknown label: all remaining classes not selected
        if want_unknown:
            for l in unknown_sources:
                for wavp in (root_path / l).glob("*.wav"):
                    rel = f"{l}/{wavp.name}"
                    if split == "train" and (rel not in val_list and rel not in test_list):
                        yield wavp, label_to_idx["unknown"]
                    elif split == "val" and rel in val_list:
                        yield wavp, label_to_idx["unknown"]
                    elif split == "test" and rel in test_list:
                        yield wavp, label_to_idx["unknown"]

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=mel_bins,
        power=2.0,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=float(top_db))

    def wav_to_mel_frames(wav_path: Path) -> np.ndarray:
        y, orig_sr = torchaudio.load(str(wav_path))  # [C, T]
        if y.dim() == 2 and y.shape[0] > 1:
            y = y.mean(dim=0, keepdim=True)
        if int(orig_sr) != int(sr):
            y = torchaudio.functional.resample(y, int(orig_sr), int(sr))
        with torch.no_grad():
            mel_power = mel_transform(y)  # [1, mel_bins, L]
            if use_log_mel:
                mel_db = amp_to_db(mel_power)
                # Keep previous behavior close to librosa power_to_db(ref=np.max, top_db=...)
                mel_db = mel_db - mel_db.amax(dim=(-2, -1), keepdim=True)
                mel_db = mel_db.clamp(min=-float(top_db), max=0.0)
                mel = (mel_db + float(top_db)) / float(top_db)
            else:
                max_val = mel_power.amax()
                if not torch.isfinite(max_val) or float(max_val) <= 0.0:
                    mel = torch.zeros_like(mel_power)
                else:
                    mel = mel_power / (max_val + 1e-8)
        return mel.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)

    def silence_mel_frames(duration_sec: float = 1.0) -> np.ndarray:
        n_frames = max(1, int(duration_sec * float(sr) / float(hop_length)))
        return np.zeros((n_frames, mel_bins), dtype=np.float32)

    class GSCMelDataset(torch.utils.data.Dataset):
        def __init__(self, split: str):
            self.items = list(iter_files(split))
            # Add synthetic silence samples by reusing the length of existing items
            if want_silence:
                # Estimate count from a small subset to avoid O(N) extra mem
                base_count = max(1, len(self.items))
                silence_count = max(1, int(float(silence_ratio) * base_count))
                self.items += [("SILENCE", label_to_idx["silence"])] * silence_count
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx: int):
            wav_path, y_idx = self.items[idx]
            if wav_path == "SILENCE":
                mel = silence_mel_frames()
            else:
                mel = wav_to_mel_frames(wav_path)
                # VAD: if energy below threshold, map to silence class
                if want_silence and vad_threshold > 0.0:
                    if float(np.mean(mel)) < float(vad_threshold):
                        mel = silence_mel_frames()
                        y_idx = label_to_idx["silence"]
            onehot = np.eye(len(labels), dtype=np.float32)[y_idx]
            return mel, onehot

    train_data = GSCMelDataset("train")
    val_data = GSCMelDataset("val")
    test_data = GSCMelDataset("test")

    dt_ms = float(hop_length) / float(sr) * 1000.0
    collate_fn = partial(
        mel_spectrogram_collate_fn,
        pad_unit=pad_unit,
        dt_ms=dt_ms,
        specaugment_prob=specaugment_prob,
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
        num_masks=num_masks,
        mixup_alpha=mixup_alpha,
        mixup_prob=mixup_prob,
    )
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=collate_fn,
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size,
        eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )

    data = Data(
        n_classes=len(labels),
        num_embeddings=mel_bins,
        train_size=len(train_data)
    )
    return train_loader, val_loader, test_loader, data


# -------------------------------
# Heidelberg Digits (HD) raw audio -> mel-spectrogram
# -------------------------------
def create_audio_hd_mel_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 0,
        seed: int = 42,
        pad_unit: int = 8192,
        mel_bins: int = 128,
        sr: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 160,
        win_length: int = 400,
        top_db: float = 80.0,
        use_log_mel: bool = True,
        root: Union[str, Path, None] = None,
        validate_on_test: bool = False,
        specaugment_prob: float = 0.0,
        freq_mask_param: int = 0,
        time_mask_param: int = 0,
        num_masks: int = 1,
        mixup_alpha: float = 0.0,
        mixup_prob: float = 0.0,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    root_path = Path(root) if root is not None else Path(cache_dir)
    audio_dir = root_path / "audio"
    train_list_path = root_path / "train_filenames.txt"
    test_list_path = root_path / "test_filenames.txt"
    assert audio_dir.exists() and audio_dir.is_dir(), f"audio dir not found at {audio_dir}"
    assert train_list_path.exists() and test_list_path.exists(), "train/test filename lists not found."

    def _read_list(p: Path) -> List[str]:
        return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

    train_rel = _read_list(train_list_path)
    test_rel = _read_list(test_list_path)

    def _label_from_rel(rel: str) -> str:
        stem = Path(rel).stem
        if "digit-" in stem:
            try:
                return stem.split("digit-")[-1].split("_")[0].split("-")[0]
            except Exception:
                pass
        parts = Path(rel).parts
        if len(parts) >= 2:
            return parts[-2]
        return stem.split('_')[0]

    labels = sorted({ _label_from_rel(rel) for rel in (train_rel + test_rel) })
    label_to_idx = {l: i for i, l in enumerate(labels)}

    def _build_items(rel_list: List[str]) -> List[Tuple[Path, int]]:
        items = []
        for rel in rel_list:
            wavp = audio_dir / rel
            if not wavp.exists():
                alt = root_path / rel
                wavp = alt if alt.exists() else wavp
            if wavp.suffix.lower() not in [".wav", ".flac", ".ogg"]:
                continue
            lab = _label_from_rel(rel)
            if lab not in label_to_idx:
                continue
            items.append((wavp, label_to_idx[lab]))
        return items

    train_items_all = _build_items(train_rel)
    test_items = _build_items(test_rel)

    if validate_on_test:
        val_items = test_items
        train_items = train_items_all
    else:
        n = len(train_items_all)
        val_len = max(1, int(0.1 * n))
        if seed is not None:
            perm = torch.randperm(n, generator=rng).tolist()
        else:
            perm = torch.randperm(n).tolist()
        train_items = [train_items_all[i] for i in perm[:-val_len]]
        val_items = [train_items_all[i] for i in perm[-val_len:]]

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=mel_bins,
        power=2.0,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=float(top_db))

    def wav_to_mel_frames(wav_path: Path) -> np.ndarray:
        y, orig_sr = torchaudio.load(str(wav_path))  # [C, T]
        if y.dim() == 2 and y.shape[0] > 1:
            y = y.mean(dim=0, keepdim=True)
        if int(orig_sr) != int(sr):
            y = torchaudio.functional.resample(y, int(orig_sr), int(sr))
        with torch.no_grad():
            mel_power = mel_transform(y)  # [1, mel_bins, L]
            if use_log_mel:
                mel_db = amp_to_db(mel_power)
                mel_db = mel_db - mel_db.amax(dim=(-2, -1), keepdim=True)
                mel_db = mel_db.clamp(min=-float(top_db), max=0.0)
                mel = (mel_db + float(top_db)) / float(top_db)
            else:
                max_val = mel_power.amax()
                if not torch.isfinite(max_val) or float(max_val) <= 0.0:
                    mel = torch.zeros_like(mel_power)
                else:
                    mel = mel_power / (max_val + 1e-8)
        return mel.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)

    class HDMelDataset(torch.utils.data.Dataset):
        def __init__(self, items: List[Tuple[Path, int]]):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx: int):
            wav_path, y_idx = self.items[idx]
            mel = wav_to_mel_frames(wav_path)
            onehot = np.eye(len(labels), dtype=np.float32)[y_idx]
            return mel, onehot

    train_data = HDMelDataset(train_items)
    val_data = HDMelDataset(val_items)
    test_data = HDMelDataset(test_items)

    dt_ms = float(hop_length) / float(sr) * 1000.0
    collate_fn = partial(
        mel_spectrogram_collate_fn,
        pad_unit=pad_unit,
        dt_ms=dt_ms,
        specaugment_prob=specaugment_prob,
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
        num_masks=num_masks,
        mixup_alpha=mixup_alpha,
        mixup_prob=mixup_prob,
    )
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=collate_fn,
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size,
        eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )

    data = Data(
        n_classes=len(labels),
        num_embeddings=mel_bins,
        train_size=len(train_data)
    )
    return train_loader, val_loader, test_loader, data


# -------------------------------
# Heidelberg Digits (HD) raw audio via audio->spike conversion
# -------------------------------
def create_events_hd_audio_snn_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 0,
        seed: int = 42,
        pad_unit: int = 8192,
        cut_mix: float = 0.0,
        mel_bins: int = 128,
        root: Union[str, Path, None] = None,
        validate_on_test: bool = False,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    Create a dataset for Heidelberg Digits raw audio where each .wav is converted
    to spike events using the same audio->spike routine as GSC2Spike.py.
    Expects the directory to contain:
      - audio/ (with subfolders per label, or files organized in subdirs)
      - train_filenames.txt, test_filenames.txt listing relative paths under audio/
    """
    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    root_path = Path(root) if root is not None else Path(cache_dir)
    audio_dir = root_path / "audio"
    train_list_path = root_path / "train_filenames.txt"
    test_list_path = root_path / "test_filenames.txt"
    assert audio_dir.exists() and audio_dir.is_dir(), f"audio dir not found at {audio_dir}"
    assert train_list_path.exists() and test_list_path.exists(), "train/test filename lists not found."

    # Read file lists and infer labels from immediate parent directory of each wav
    def _read_list(p: Path) -> List[str]:
        return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

    train_rel = _read_list(train_list_path)
    test_rel = _read_list(test_list_path)
    all_rel = train_rel + test_rel

    def _label_from_rel(rel: str) -> str:
        # HD audio filenames encode the digit label, e.g. "..._digit-7.flac"
        stem = Path(rel).stem
        # Try to parse "digit-<n>" from the filename
        if "digit-" in stem:
            try:
                return stem.split("digit-")[-1].split("_")[0].split("-")[0]
            except Exception:
                pass
        # Fallback: use parent directory if available
        parts = Path(rel).parts
        if len(parts) >= 2:
            return parts[-2]
        # Last resort: take the prefix before the first underscore
        return stem.split('_')[0]

    labels = sorted({ _label_from_rel(rel) for rel in all_rel })
    label_to_idx = {l: i for i, l in enumerate(labels)}

    def _build_items(rel_list: List[str]) -> List[Tuple[Path, int]]:
        items = []
        for rel in rel_list:
            wavp = audio_dir / rel
            if not wavp.exists():
                # Try without leading 'audio/' if present in list
                alt = root_path / rel
                wavp = alt if alt.exists() else wavp
            if wavp.suffix.lower() not in [".wav", ".flac", ".ogg"]:
                continue
            lab = _label_from_rel(rel)
            if lab not in label_to_idx:
                continue
            items.append((wavp, label_to_idx[lab]))
        return items

    train_items_all = _build_items(train_rel)
    test_items = _build_items(test_rel)

    # Build validation split from train if requested
    if validate_on_test:
        val_items = test_items
        train_items = train_items_all
    else:
        n = len(train_items_all)
        val_len = max(1, int(0.1 * n))
        if seed is not None:
            perm = torch.randperm(n, generator=rng).tolist()
        else:
            perm = torch.randperm(n).tolist()
        val_idx = set(perm[-val_len:])
        train_items = [train_items_all[i] for i in perm[:-val_len]]
        val_items = [train_items_all[i] for i in perm[-val_len:]]

    class HDAudioSNNDataset(torch.utils.data.Dataset):
        def __init__(self, items: List[Tuple[Path, int]]):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx: int):
            from GSC2Spike import wav_to_spike_events
            wav_path, y_idx = self.items[idx]
            events = wav_to_spike_events(str(wav_path), mel_bins=mel_bins)
            onehot = np.eye(len(labels), dtype=np.float32)[y_idx]
            return events, onehot

    train_data = HDAudioSNNDataset(train_items)
    val_data = HDAudioSNNDataset(val_items)
    test_data = HDAudioSNNDataset(test_items)

    collate_fn = partial(
        event_stream_collate_fn,
        resolution=(mel_bins,),
        pad_unit=pad_unit,
        no_time_information=False
    )
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix),
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size,
        eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )

    data = Data(
        n_classes=len(labels),
        num_embeddings=mel_bins,
        train_size=len(train_data)
    )
    return train_loader, val_loader, test_loader, data
# Register all dataset creation functions in a dictionary for easy access
Datasets = {
    "shd-classification": create_events_shd_classification_dataset,
    "ssc-classification": create_events_ssc_classification_dataset,
    "dvs-gesture-classification": create_events_dvs_gesture_classification_dataset,
    "dvs-lip-classification": create_events_dvs_lip_classification_dataset,
    "gsc-snn-classification": create_events_gsc_snn_classification_dataset,
    "hd-audio-snn-classification": create_events_hd_audio_snn_classification_dataset,
    "gsc-mel-classification": create_audio_gsc_mel_classification_dataset,
    "hd-audio-mel-classification": create_audio_hd_mel_classification_dataset,
}
