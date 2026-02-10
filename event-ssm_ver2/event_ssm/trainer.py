import time
import os
import json
import sys
import wandb
from collections import defaultdict, OrderedDict
from omegaconf import OmegaConf as om
from omegaconf import DictConfig
import jax.numpy as jnp
import jax.random
from jaxtyping import Array
from typing import Callable, Dict, Optional, Iterator, Any
from flax.training.train_state import TrainState
from flax.training import checkpoints
from flax import jax_utils
from functools import partial

# -------------------------
# 以下为调试工具函数，可以单独放到 utils/debug_utils.py 中
# -------------------------
import json
from flax import traverse_util
import jax.numpy as jnp

def get_stats(x):
    """计算单个数组的统计数据。"""
    return {
        "mean": float(jnp.mean(x)),
        "std": float(jnp.std(x)),
        "var": float(jnp.var(x)),
        "min": float(jnp.min(x)),
        "max": float(jnp.max(x)),
        "shape": x.shape,
    }

def compute_tree_stats(tree):
    """
    将参数树扁平化后，返回一个字典，键为参数名（路径），值为统计数据。
    修改：如果 key 已经是字符串则直接使用，否则拼接元组中的各部分。
    """
    flat_dict = traverse_util.flatten_dict(tree, sep='/')
    stats_dict = {}
    for key, value in flat_dict.items():
        if isinstance(key, tuple):
            key_str = "/".join(key)
        else:
            key_str = key
        stats_dict[key_str] = get_stats(value)
    return stats_dict

def print_param_statistics(params, log_to_wandb=False):
    """
    计算并打印参数的统计数据，也可以通过 wandb 记录。
    """
    stats = compute_tree_stats(params)
    print("==== 模型参数统计 ====")
    print(json.dumps(stats, indent=2))
    if log_to_wandb:
        import wandb
        wandb.log({"model_parameters_stats": stats})

def log_debug_statistics(debug_info, log_to_wandb=False):
    """
    对传入的字典 debug_info 中的各个变量计算统计数据并打印。
    """
    stats = {}
    for name, value in debug_info.items():
        stats[name] = {
            "mean": float(jnp.mean(value)),
            "std": float(jnp.std(value)),
            "var": float(jnp.var(value)),
            "min": float(jnp.min(value)),
            "max": float(jnp.max(value)),
            "shape": value.shape,
        }
    print("==== 模型中间变量统计 ====")
    print(json.dumps(stats, indent=2))
    if log_to_wandb:
        import wandb
        wandb.log({"model_debug_stats": stats})

# -------------------------
# 以下为 TrainerModule 及其相关函数
# -------------------------

@partial(jax.jit, static_argnums=(1,))
def reshape_batch_per_device(x, num_devices):
    return jax.tree_util.tree_map(partial(reshape_array_per_device, num_devices=num_devices), x)

def reshape_array_per_device(x, num_devices):
    # 计算能被整除的最大样本数
    new_size = (x.shape[0] // num_devices) * num_devices
    if new_size < x.shape[0]:
        print(f"[WARNING] Dropping last {x.shape[0] - new_size} samples to make batch size divisible by {num_devices}")
        x = x[:new_size]
    batch_size_per_device = new_size // num_devices
    return x.reshape((num_devices, batch_size_per_device) + x.shape[1:])

class TrainerModule:
    """
    Handles training and logging of models. Most of the boilerplate code is hidden from the user.
    """
    def __init__(
            self,
            train_state: TrainState,
            training_step_fn: Callable,
            evaluation_step_fn: Callable,
            world_size: int,
            config: DictConfig,
    ):
        """
        :param train_state: A TrainState object that contains the model parameters, optimizer states etc.
        :param training_step_fn: A function that takes the train_state and a batch of data and returns the updated train_state and metrics.
        :param evaluation_step_fn: A function that takes the train_state and a batch of data and returns the updated train_state and metrics.
        :param world_size: Number of devices to run the training on.
        :param config: The configuration of the training run.
        """
        super().__init__()
        self.train_state = train_state
        self.train_step = training_step_fn
        self.eval_step = evaluation_step_fn

        self.world_size = world_size
        self.log_config = config.logging
        self.epoch_idx = 0
        self.num_epochs = config.training.num_epochs
        self.best_eval_metrics = {}

        # logger details
        self.log_dir = os.path.join(self.log_config.log_dir)
        print('[*] Logging to', self.log_dir)

        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.isdir(os.path.join(self.log_dir, 'metrics')):
            os.makedirs(os.path.join(self.log_dir, 'metrics'))
        if not os.path.isdir(os.path.join(self.log_dir, 'checkpoints')):
            os.makedirs(os.path.join(self.log_dir, 'checkpoints'))

        num_parameters = int(sum(
            [arr.size for arr in jax.tree_util.tree_flatten(self.train_state.params)[0]
             if isinstance(arr, Array)]
        ) / self.world_size)
        print("[*] Number of model parameters:", num_parameters)

        if self.log_config.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                dir=self.log_config.log_dir,
                project="test",
                # entity="event-ssm",
                config=om.to_container(config, resolve=True))
            wandb.config.update({'SLURM_JOB_ID': os.getenv('SLURM_JOB_ID')})

            # log number of parameters
            wandb.run.summary['Num parameters'] = num_parameters
            wandb.define_metric(self.log_config.summary_metric, summary='max')

    def train_model(
            self,
            train_loader: Iterator,
            val_loader: Iterator,
            dropout_key: Array,
            test_loader: Optional[Iterator] = None,
            # wandb_log: object=None,
    ) -> Dict[str, Any]:
        """
        Trains a model on a dataset.
        :param train_loader: Data loader of the training set.
        :param val_loader: Data loader of the validation set.
        :param dropout_key: Random key for dropout.
        :param test_loader: Data loader of the test set.
        :return: A dictionary of the best evaluation metrics.
        """

        # Prepare training loop
        self.on_training_start()

        for epoch_idx in range(1, self.num_epochs+1):
            self.epoch_idx = epoch_idx

            # run training step for this epoch
            train_metrics = self.train_epoch(train_loader, dropout_key)

            self.on_training_epoch_end(train_metrics)

            # Validation every epoch
            eval_metrics = self.eval_model(
                val_loader,
                log_prefix='Performance/Validation',
            )
            # wandb_log.log({"val_acc": eval_metrics["Performance/Validation accuracy"], 
                        #    "loss"   : eval_metrics["Performance/Validation loss"]})

            self.on_validation_epoch_end(eval_metrics)

            if self.log_config.wandb:
                from optax import MultiStepsState
                wandb_metrics = {'Performance/epoch': epoch_idx}
                wandb_metrics.update(train_metrics)
                wandb_metrics.update(eval_metrics)
                if isinstance(self.train_state.opt_state, MultiStepsState):
                    lr = self.train_state.opt_state.inner_opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'].mean().item()
                else:
                    lr = self.train_state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'].item()
                wandb_metrics['learning rate'] = lr
                wandb.log(wandb_metrics)

        # Test best model if possible
        if test_loader is not None:
            self.load_model()
            test_metrics = self.eval_model(
                test_loader,
                log_prefix='Performance/Test',
            )
            self.save_metrics('test', test_metrics)
            self.best_eval_metrics.update(test_metrics)

            if self.log_config.wandb:
                wandb.log(test_metrics)

            print('-' * 89)
            print('| End of Training |')
            print('| Test  Metrics |',
                  ' | '.join([f"{k.split('/')[1].replace('Test','')}: {v:5.2f}" 
                               for k, v in test_metrics.items() if 'Test' in k]))
            print('-' * 89)

        return self.best_eval_metrics

    def train_epoch(self, train_loader: Iterator, dropout_key) -> Dict[str, Any]:
        """
        Trains the model on one epoch of the training set.
        :param train_loader: Data loader of the training set.
        :param dropout_key: Random key for dropout.
        :return: A dictionary of the training metrics.
        """

        # Train model for one epoch, and log avg loss and accuracy
        metrics = defaultdict(float)
        running_metrics = defaultdict(float)
        num_batches = 0
        num_train_batches = len(train_loader)
        start_time = time.time()
        epoch_start_time = start_time

        # set up intra epoch logging
        log_interval = self.log_config.interval

        for i, batch in enumerate(train_loader):
            num_batches += 1

            # skip batches with empty sequences which might randomly occur due to data augmentation
            _, _, _, lengths = batch
            if jnp.any(lengths == 0):
                continue

            if self.world_size > 1:
                step_key, dropout_key = jax.vmap(jax.random.split, in_axes=0, out_axes=1)(dropout_key)
                step_key = jax.vmap(jax.random.fold_in)(step_key, jnp.arange(self.world_size))
                batch = reshape_batch_per_device(batch, self.world_size)
            else:
                step_key, dropout_key = jax.random.split(dropout_key)

            self.train_state, step_metrics = self.train_step(self.train_state, batch, step_key)

            # exit from training if loss is nan
            if jnp.isnan(step_metrics['loss']).any():
                print("EXITING TRAINING DUE TO NAN LOSS")
                sys.exit(1)

            # record metrics
            for key in step_metrics:
                metrics['Performance/Training ' + key] += step_metrics[key]
                running_metrics['Performance/Training ' + key] += step_metrics[key]

            # print metrics to terminal
            if (i + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                start_time = time.time()
                print(f'| epoch {self.epoch_idx} | {i + 1}/{num_train_batches} batches | ms/batch {elapsed * 1000 / log_interval:5.2f} |',
                      ' | '.join([f'{k}: {jnp.mean(v).item() / log_interval:5.2f}' for k, v in running_metrics.items()]))
                for key in step_metrics:
                    running_metrics['Performance/Training ' + key] = 0

        metrics = {key: jnp.mean(metrics[key] / num_batches).item() for key in metrics}
        metrics['epoch_time'] = time.time() - epoch_start_time
        return metrics

    def eval_model(
            self,
            data_loader: Iterator,
            log_prefix: Optional[str] = '',
    ) -> Dict[str, Any]:
        """
        Evaluates the model on a dataset.
        :param data_loader: Data loader of the dataset.
        :param log_prefix: Prefix to add to the keys of the logged metrics such as "Best" or "Validation".
        :return: A dictionary of the evaluation metrics.
        """

        # Test model on all images of a data loader and return avg loss
        metrics = defaultdict(float)
        num_batches = 0

        for i, batch in enumerate(iter(data_loader)):
            if self.world_size > 1:
                batch = reshape_batch_per_device(batch, self.world_size)

            self.train_state, step_metrics = self.eval_step(self.train_state, batch)

            for key in step_metrics:
                metrics[key] += step_metrics[key]
            num_batches += 1

        prefix = log_prefix + ' ' if log_prefix else ''
        metrics = {(prefix + key): jnp.mean(metrics[key] / num_batches).item() for key in metrics}
        return metrics

    def is_new_model_better(self, new_metrics: Dict[str, Any], old_metrics: Dict[str, Any]) -> bool:
        """
        Compares two sets of evaluation metrics to decide whether the
        new model is better than the previous ones or not.
        :params new_metrics: A dictionary of the evaluation metrics of the new model.
        :params old_metrics: A dictionary of the evaluation metrics of the previously
            best model, i.e. the one to compare to.
        :return: True if the new model is better than the old one, and False otherwise.
        """
        if len(old_metrics) == 0:
            return True
        for key, is_larger in [('val/val_metric', False), ('Performance/Validation accuracy', True), ('Performance/Validation loss', False)]:
            if key in new_metrics:
                if is_larger:
                    return new_metrics[key] > old_metrics[key]
                else:
                    return new_metrics[key] < old_metrics[key]
        assert False, f'No known metrics to log on: {new_metrics}'

    def save_metrics(self, filename: str, metrics: Dict[str, Any]):
        """
        Saves a dictionary of metrics to file. Can be used as a textual
        representation of the validation performance for checking in the terminal.
        :param filename: The name of the file to save the metrics to.
        :param metrics: A dictionary of the metrics to save.
        """
        with open(os.path.join(self.log_dir, f'metrics/{filename}.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

    def save_model(self):
        """
        Saves the model to a file. The model is saved in the log directory.
        """
        if self.world_size > 1:
            state = jax_utils.unreplicate(self.train_state)
        else:
            state = self.train_state
        checkpoints.save_checkpoint(
            ckpt_dir=os.path.abspath(os.path.join(self.log_dir, 'checkpoints')),
            target=state,
            step=state.step,
            overwrite=True,
            keep=1
        )
        del state

    def load_model(self):
        """
        Loads the model from a file. The model is loaded from the log directory.
        """
        ckpt_dir = os.path.abspath(os.path.join(self.log_dir, 'checkpoints'))
        if self.world_size > 1:
            state = jax_utils.unreplicate(self.train_state)
            raw_restored = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state)
            self.train_state = jax_utils.replicate(raw_restored)
            del state
        else:
            self.train_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=self.train_state)

    def on_training_start(self):
        """
        Method called before training is started. Can be used for additional
        initialization operations etc.
        """
        pass

    def on_training_epoch_end(self, train_metrics):
        """
        Method called at the end of each training epoch. Can be used for additional logging or similar.
        """
        print('-' * 89)
        print(f"| end of epoch {self.epoch_idx:3d} | time per epoch: {train_metrics['epoch_time']:5.2f}s |")
        print('| Train Metrics |', ' | '.join(
            [f"{k.split('/')[1].replace('Training ', '')}: {v:5.2f}" 
             for k, v in train_metrics.items() if 'Train' in k]))

        # check metrics for nan values and possibly exit training
        if jnp.isnan(train_metrics['Performance/Training loss']).item():
            print("EXITING TRAINING DUE TO NAN LOSS")
            sys.exit(1)

        # 每隔10个epoch打印一次模型参数的统计信息
        if self.epoch_idx % 10 == 0:
            if self.world_size > 1:
                # 对于多卡训练，取第一个设备的参数
                params_single = jax.tree_util.tree_map(lambda x: x[0], self.train_state.params)
            else:
                params_single = self.train_state.params

            all_stats = compute_tree_stats(params_single)
            # 分组统计：根据键名判断参数类型
            lambda_stats = {k: v for k, v in all_stats.items() if 'Lambda' in k}
            B_stats = {k: v for k, v in all_stats.items() if ('B' in k) and ('Lambda' not in k)}
            C_stats = {k: v for k, v in all_stats.items() if 'C' in k}
            D_stats = {k: v for k, v in all_stats.items() if 'D' in k}
            grouped_keys = set(lambda_stats.keys()).union(set(B_stats.keys())).union(set(C_stats.keys())).union(set(D_stats.keys()))
            other_stats = {k: v for k, v in all_stats.items() if k not in grouped_keys}

            print("==== Parameter Statistics: Lambda ====")
            print(json.dumps(lambda_stats, indent=2))
            print("==== Parameter Statistics: B Matrices ====")
            print(json.dumps(B_stats, indent=2))
            print("==== Parameter Statistics: C Matrices ====")
            print(json.dumps(C_stats, indent=2))
            print("==== Parameter Statistics: D Matrices ====")
            print(json.dumps(D_stats, indent=2))
            print("==== Parameter Statistics: Other ====")
            print(json.dumps(other_stats, indent=2))

    def on_validation_epoch_end(self, eval_metrics: Dict[str, Any]):
        """
        Method called at the end of each validation epoch. Can be used for additional logging and evaluation.
        Args:
          eval_metrics: A dictionary of the validation metrics. New metrics added to this dictionary will be logged as well.
        """
        print('| Eval  Metrics |', ' | '.join(
            [f"{k.split('/')[1].replace('Validation ', '')}: {v:5.2f}" 
             for k, v in eval_metrics.items() if 'Validation' in k]))
        print('-' * 89)

        self.save_metrics(f'eval_epoch_{str(self.epoch_idx).zfill(3)}', eval_metrics)

        # Save best model
        if self.is_new_model_better(eval_metrics, self.best_eval_metrics):
            self.best_eval_metrics = eval_metrics
            self.save_model()
            self.save_metrics('best_eval', eval_metrics)
