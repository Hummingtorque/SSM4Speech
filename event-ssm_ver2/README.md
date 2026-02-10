# Scalable Event-by-event Processing of Neuromorphic Sensory Signals With Deep State-Space Models
![Figure 1](docs/figure1.png)
This is the official implementation of our paper [Scalable Event-by-event Processing of Neuromorphic Sensory Signals With Deep State-Space Models
](https://arxiv.org/abs/2404.18508).
The core motivation for this work was the irregular time-series modeling problem presented in the paper [Simplified State Space Layers for Sequence Modeling
](https://arxiv.org/abs/2208.04933). 
We acknowledge the awesome [S5 project](https://github.com/lindermanlab/S5) and the trainer class provided by this [UvA tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide4/Research_Projects_with_JAX.html), which highly influenced our code.

Our project treats a quite general machine learning problem:
Modeling **long sequences** that are **irregularly** sampled by a possibly large number of **asynchronous** sensors.
This problem is particularly present in the field of neuromorphic computing, where event-based sensors emit up to millions events per second from asynchronous channels.

We show how linear state-space models can be tuned to effectively model asynchronous event-based sequences.
Our contributions are
- Integration of dirac delta coded event streams
- time-invariant input normalization to effectively learn from long event-streams
- formulating neuromorphic event-streams as a language modeling problem with **asynchronous tokens**
- effectively model event-based vision **without frames and without CNNs** 

## Quick guide (PyTorch training/eval in this folder)

This subdirectory contains a PyTorch training/evaluation workflow for event-stream classification.
Use this section to navigate the code and reproduce/evaluate runs quickly.

- **Entrypoints**
  - `run_training.py`: builds dataloaders, constructs the model (`ClassificationModel` with `TorchS5` blocks), trains with warmup/cosine schedule and optional EMA, saves best checkpoint as `outputs/.../checkpoints/model.pt`, prints validation each epoch and final test accuracy.
  - `run_evaluation.py`: loads the same dataset pipeline and a checkpoint (argument `checkpoint=<path>`), then evaluates on the test set.

- **Data loading and preprocessing**
  - `event_ssm/dataloading.py`:
    - `Datasets`: registry mapping task names to dataset builders:
      - `"shd-classification"` (Spiking Heidelberg Digits, 20 classes)
      - `"ssc-classification"` (Spiking Speech Commands, 35 classes)
      - `"dvs-gesture-classification"` (DVS Gesture, 11 classes)
      - `"dvs-lip-classification"` (DVS-Lip, 100 classes)
    - For SHD, `create_events_shd_classification_dataset(...)` uses `tonic.datasets.SHD`,
      applies train-time augmentations, and creates a 90/10 train/val split (unless `training.validate_on_test=true`).
    - `event_stream_collate_fn(...)` converts raw events to model inputs:
      `tokens` (B, L), `timesteps` (B, L, in ms), and `lengths` (B). Validation/test use evaluation collate (no cut-mix).
  - `event_ssm/transform.py`: custom transforms used in training (e.g., `DropEventChunk`, `Jitter1D`), and label utilities (`OneHotLabels`), as well as optional cut-mix.

- **Model definition**
  - `event_ssm/seq_model.py`:
    - `ClassificationModel`: end-to-end classifier; forward is `model(x, timesteps, lengths, train=False)`.
    - `StackedEncoderModel`: token embedding + stacked `SequenceStage`s + temporal pooling; total downsampling depends on `pooling_stride` and number of stages.
  - `event_ssm/layers.py`:
    - `TorchS5`: diagonal SSM block with on-the-fly discretization (`'zoh' | 'dirac' | 'async'`) and optional stride pooling (`EventPooling`).
    - `SequenceLayer`/`SequenceStage`: stacks SSM + lightweight gating/normalization layers and handles residual/pooling.

- **Configuration**
  - Hydra is used throughout. For SHD medium, see `configs/model/shd/medium.yaml` (e.g., `d_model=128`, `pooling_stride=8`, `classification_mode=pool`, etc.).
  - Each run also writes its resolved config to `outputs/<timestamp>/config.yaml`.

- **Artifacts from a sample run**
  - Checkpoint: `outputs/2025-12-02-00-58-29/checkpoints/model.pt`
  - Run config: `outputs/2025-12-02-00-58-29/config.yaml`
  - Current best run:
    - Log: `SSMnoJax/event-ssm_ver2/slurm_logs/train_37069531_ReBu97ct.out`
    - Checkpoint dir: `outputs/2025-12-02-00-58-29_BestSHD97ct`

- **Example commands (local)**
  - Train (SHD, medium):
    ```bash
    python run_training.py \
      task.name=shd-classification model=shd/medium \
      data_dir=/path/to/data \
      training.num_epochs=180 training.validate_on_test=false
    ```
  - Evaluate on test set:
    ```bash
    python run_evaluation.py \
      task.name=shd-classification model=shd/medium \
      data_dir=/path/to/data \
      checkpoint=/path/to/outputs/2025-12-02-00-58-29/checkpoints/model.pt
    ```
  - Notes: You can adjust batch sizes via `training.per_device_batch_size` and `training.per_device_eval_batch_size` if memory-constrained.

