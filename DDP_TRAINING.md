# PyTorch DDP 多卡训练指南

本项目已经支持 PyTorch Distributed Data Parallel (DDP) 多卡训练，可以大幅提升训练速度。

## 主要修改

### 1. 训练脚本修改 (`train.py`)
- 添加了分布式训练参数支持
- 修改了模型初始化以支持 DDP
- 添加了分布式采样器 (DistributedSampler)
- 确保只在主进程进行日志记录和模型保存
- 修复了设备管理和同步逻辑

### 2. 数据加载器修改 (`utils.py`)
- 在分布式模式下使用 DistributedSampler
- 自动处理数据分片和打乱

## 使用方法

### 方法一：使用 torchrun (推荐，PyTorch 1.10+)
```bash
# 2卡训练
bash train_torchrun.sh

# 或者手动指定参数
torchrun --nproc_per_node=2 --standalone train.py --distributed [其他参数]

# 4卡训练
torchrun --nproc_per_node=4 --standalone train.py --distributed [其他参数]
```

### 方法二：使用 torch.distributed.launch (兼容旧版本)
```bash
# 2卡训练  
bash train_ddp.sh

# 或者手动指定参数
python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --distributed [其他参数]
```

### 方法三：单卡训练 (保持兼容)
```bash
# 不使用 --distributed 参数即为单卡训练
python train.py --layers 50 --model_type res50_ASPP_lorm [其他参数]
```

## 参数说明

### 新增的分布式参数
- `--distributed`: 启用分布式训练 (必须)
- `--world_size`: 总进程数 (自动设置)
- `--local_rank`: 本地GPU rank (自动设置)
- `--dist_url`: 分布式通信URL (默认 'env://')

### 重要配置建议
- `--batchsize`: 设置为每GPU的batch size，总batch size = GPU数量 × 每GPU batch size
- `--workers`: 数据加载器的worker数量，建议设置为4-8
- `--lr`: 学习率会自动根据GPU数量进行缩放

## 性能对比

### 单卡 vs 多卡
- **单卡**: batch_size=16, lr=1e-3
- **双卡**: batch_size=8×2=16, lr=2e-3 (自动缩放)
- **四卡**: batch_size=4×4=16, lr=4e-3 (自动缩放)

### 预期加速比
- 2卡: ~1.7-1.9x 加速
- 4卡: ~3.2-3.6x 加速
- 8卡: ~6-7x 加速

## 注意事项

### 1. 环境要求
- PyTorch >= 1.6.0 (建议 >= 1.10.0)
- CUDA 和 NCCL 支持
- 多GPU环境

### 2. 数据集要求
- 确保数据集足够大，能够有效分片
- 每个GPU至少要有足够的数据进行训练

### 3. 内存和显存
- 每张卡的显存使用量基本不变
- 总内存使用量会增加 (每个进程独立)

### 4. 调试技巧
```bash
# 检查GPU数量
nvidia-smi

# 设置环境变量进行调试
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

# 单进程调试
python train.py --distributed --local_rank=0 [其他参数]
```

### 5. 常见错误解决
- **NCCL timeout**: 增加 `NCCL_TIMEOUT` 环境变量
- **端口占用**: 修改 `--master_port` 参数
- **OOM错误**: 减少每GPU的batch size

## VS Code 调试配置

已更新 `.vscode/launch.json`，添加了 DDP 训练的调试配置：

```json
{
    "name": "Debug DDP Training",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/train.py",
    "console": "integratedTerminal",
    "args": [
        "--distributed",
        "--local_rank", "0",
        "--layers", "50",
        "--model_type", "res50_ASPP_lorm",
        // ... 其他参数
    ],
    "env": {
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12355"
    }
}
```

## 性能监控

### 1. TensorBoard
```bash
tensorboard --logdir=./log
```

### 2. GPU监控
```bash
# 实时监控GPU使用率
watch -n 1 nvidia-smi

# 或使用 gpustat
pip install gpustat
gpustat -i 1
```

### 3. 网络监控 (多机训练)
```bash
# 监控网络带宽
iftop -i [网卡名称]
```

## 最佳实践

1. **批大小选择**: 总batch size保持不变，按GPU数量分配
2. **学习率调整**: 使用线性缩放规则或warm-up策略  
3. **同步频率**: 使用默认的每步同步，避免手动同步
4. **模型保存**: 只在主进程 (rank 0) 保存模型
5. **日志输出**: 只在主进程输出日志和TensorBoard记录
6. **随机种子**: 确保所有进程使用相同的随机种子

开始多卡训练后，您应该能看到显著的训练速度提升！