import os
import json
import jax
import numpy as np
from omegaconf import OmegaConf, open_dict
from flax.training import checkpoints
from flax import traverse_util

# --------------------------------------------------------------------------
# 从您的项目代码中正确导入模块
# --------------------------------------------------------------------------
from event_ssm.dataloading import Datasets
from event_ssm.ssm import init_S5SSM
from event_ssm.seq_model import BatchClassificationModel
from event_ssm.train_utils import init_model_state, TrainState

def main():
    """
    主函数，用于加载模型并将其参数保存到文件中。
    """
    print("[*] 开始加载模型参数...")

    # 1. 设置路径
    ckpt_dir = '/nfs/turbo/coe-wluee/zxygo/event-ssm_ver2/outputs/2025-02-14-23-00-34_DVS128_best/checkpoints'
    config_path = os.path.join(os.path.dirname(ckpt_dir), 'config.yaml')
    
    if not os.path.exists(ckpt_dir) or not os.path.exists(config_path):
        print(f"[错误] 路径不存在。请检查: {ckpt_dir} 和 {config_path}")
        return
        
    print(f"[*] 从配置文件加载设置: {config_path}")
    print(f"[*] 从检查点加载模型: {ckpt_dir}")

    # 2. 加载配置并准备环境
    cfg = OmegaConf.load(config_path)
    key = jax.random.PRNGKey(cfg.seed)
    
    # 3. 加载数据集
    print("[*] 正在加载数据集以获取输入形状和元信息...")
    task_name = cfg.task.name 
    create_dataset_fn = Datasets[task_name]
    train_loader, _, _, data = create_dataset_fn(
        cache_dir=cfg.data_dir,
        seed=cfg.seed,
        world_size=1,
        **cfg.training
    )
    dummy_batch = next(iter(train_loader))
    inputs, _, timesteps, lengths = dummy_batch
    print("[*] 数据集形状获取成功。")

    # 3.5. 动态计算并添加所有缺失的优化器参数
    print("[*] 正在动态计算优化器参数...")
    with open_dict(cfg):
        accumulation_steps = cfg.optimizer.get('accumulation_steps', 1)
        
        # 计算 total_steps 和 warmup_steps
        cfg.optimizer.total_steps = (
            cfg.training.num_epochs * len(train_loader) // accumulation_steps
        )
        cfg.optimizer.warmup_steps = (
            cfg.optimizer.warmup_epochs * len(train_loader) // accumulation_steps
        )
        
        # --- 开始新增修改 ---
        # 计算最终的学习率 ssm_lr
        # 在读取参数的脚本中，我们假设只在单个设备上运行，所以 num_devices = 1
        num_devices = 1 
        cfg.optimizer.ssm_lr = (
            cfg.optimizer.ssm_base_lr 
            * cfg.training.per_device_batch_size 
            * num_devices 
            * accumulation_steps
        )
        # --- 结束新增修改 ---

    print(f"[*] 计算得到的 total_steps: {cfg.optimizer.total_steps}")
    print(f"[*] 计算得到的 warmup_steps: {cfg.optimizer.warmup_steps}")
    print(f"[*] 计算得到的 ssm_lr: {cfg.optimizer.ssm_lr}")

    # 4. 初始化模型和"模板"状态
    print("[*] 正在初始化模型结构...")
    ssm_init_fn = init_S5SSM(**cfg.model.ssm_init)
    model = BatchClassificationModel(
        ssm=ssm_init_fn,
        num_classes=data.n_classes,
        num_embeddings=data.num_embeddings,
        **cfg.model.ssm,
    )
    single_bsz = cfg.training.per_device_batch_size
    state_template = init_model_state(
        key, model, inputs[:single_bsz], timesteps[:single_bsz], lengths[:single_bsz], cfg.optimizer
    )
    print("[*] 模型模板状态创建成功。")

    # 5. 从 Checkpoint 恢复模型状态
    print("[*] 正在从文件恢复权重...")
    restored_state = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=state_template
    )
    if restored_state is None:
        print("[错误] 加载模型失败，请检查路径和 checkpoint 文件。")
        return
    print("[*] 权重加载成功！")

    # 6. 将所有参数格式化并保存到文件
    output_filename = 'checkpoint_parameters.json'
    print(f"[*] 正在将参数写入文件: {output_filename} ...")
    
    flat_params = traverse_util.flatten_dict(restored_state.params)
    params_for_json = {}
    for path, array in flat_params.items():
        key_str = "/".join(path)
        params_for_json[key_str] = jax.device_get(array).tolist()

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(params_for_json, f, indent=2)
    
    print("\n" + "="*80)
    print(f"[*] 任务完成。参数已成功保存到文件: {os.path.abspath(output_filename)}")


if __name__ == '__main__':
    main()