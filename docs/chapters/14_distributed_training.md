# 分布式训练策略

## 分布式训练概述

大模型训练需要将计算分布到多个 NPU 上，主要有三种并行策略：

```
┌─────────────────────────────────────────────────────────┐
│                   并行策略                               │
│                                                         │
│  数据并行（DP）    模型并行（MP）    流水线并行（PP）      │
│  每卡完整模型      模型按层分割      模型按阶段分割        │
│  数据分割          数据完整          数据完整             │
│  梯度 AllReduce    激活 AllReduce    微批次流水            │
└─────────────────────────────────────────────────────────┘
```

---

## 数据并行（Data Parallelism）

最简单的并行策略，每个 NPU 持有完整的模型副本，处理不同的数据批次。

```
NPU0: 模型副本 + 数据[0:N/4]  ─┐
NPU1: 模型副本 + 数据[N/4:N/2] ─┤─→ AllReduce 梯度 ─→ 更新参数
NPU2: 模型副本 + 数据[N/2:3N/4]─┤
NPU3: 模型副本 + 数据[3N/4:N]  ─┘
```

**适用场景**：模型能放入单卡内存，需要提升吞吐量

### PyTorch DDP 实现

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_npu

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("hccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    # 设置设备
    device = torch.device(f'npu:{rank}')
    torch.npu.set_device(device)
    
    # 创建模型并移到 NPU
    model = MyModel().to(device)
    
    # 包装为 DDP
    ddp_model = DDP(model, device_ids=[rank])
    
    # 数据加载（每个进程加载不同数据）
    dataset = MyDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # 每个 epoch 重新打乱
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # 自动 AllReduce 梯度
            optimizer.step()
    
    cleanup()

# 启动
if __name__ == "__main__":
    world_size = 8
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
```

---

## 张量并行（Tensor Parallelism）

将单个算子（如矩阵乘法）的计算分布到多个 NPU 上。

```
# 列并行（Column Parallel）
输入 X [B, H] ─→ 分割权重 W 的列
NPU0: X × W[:, 0:H/2] = Y0 [B, H/2]
NPU1: X × W[:, H/2:H] = Y1 [B, H/2]
→ AllGather → Y [B, H]

# 行并行（Row Parallel）
输入 X [B, H] ─→ 分割输入和权重的行
NPU0: X[:, 0:H/2] × W[0:H/2, :] = Y0 [B, H]
NPU1: X[:, H/2:H] × W[H/2:H, :] = Y1 [B, H]
→ AllReduce → Y [B, H]
```

**适用场景**：单层参数量超过单卡内存（如大型 Transformer 的 FFN 层）

### Megatron-LM 风格张量并行

```python
# 列并行线性层
class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, tp_size):
        super().__init__()
        self.tp_size = tp_size
        # 每个 NPU 只持有 out_features/tp_size 列
        self.weight = torch.nn.Parameter(
            torch.randn(out_features // tp_size, in_features)
        )
    
    def forward(self, x):
        # 本地矩阵乘
        local_out = torch.nn.functional.linear(x, self.weight)
        # AllGather 收集所有 NPU 的输出
        output = all_gather(local_out, dim=-1)
        return output
```

---

## 流水线并行（Pipeline Parallelism）

将模型按层分割到不同 NPU，形成流水线。

```
NPU0: Layer 1-4   ─→ 激活值 ─→ NPU1: Layer 5-8
                                      ─→ 激活值 ─→ NPU2: Layer 9-12
                                                        ─→ 激活值 ─→ NPU3: Layer 13-16
```

**微批次（Micro-batch）流水线**：

```
时间轴：
NPU0: [mb0 fwd] [mb1 fwd] [mb2 fwd] [mb3 fwd] [mb3 bwd] [mb2 bwd] [mb1 bwd] [mb0 bwd]
NPU1:           [mb0 fwd] [mb1 fwd] [mb2 fwd] [mb3 fwd] [mb3 bwd] [mb2 bwd] [mb1 bwd] [mb0 bwd]
NPU2:                     [mb0 fwd] [mb1 fwd] [mb2 fwd] [mb3 fwd] [mb3 bwd] [mb2 bwd] [mb1 bwd] [mb0 bwd]
```

**适用场景**：模型层数多，单层参数量不大

---

## 3D 并行（DP + TP + PP）

大模型训练通常组合使用三种并行策略：

```
总 NPU 数 = DP × TP × PP

例：64 卡 = 8(DP) × 4(TP) × 2(PP)
- 8 个数据并行组，每组 8 卡
- 每组内 4 卡做张量并行
- 2 个流水线阶段
```

---

## ZeRO 优化器

ZeRO（Zero Redundancy Optimizer）通过分片存储优化器状态、梯度和参数来减少内存占用：

| ZeRO 阶段 | 分片内容 | 内存节省 |
|-----------|---------|---------|
| ZeRO-1 | 优化器状态 | ~4× |
| ZeRO-2 | 优化器状态 + 梯度 | ~8× |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 | ~64× |

### DeepSpeed ZeRO 配置

```json
{
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000
    },
    "train_batch_size": 256,
    "gradient_accumulation_steps": 4
}
```

---

## 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    # 自动混合精度（FP16 前向，FP32 梯度）
    with autocast(device_type='npu', dtype=torch.float16):
        output = model(batch)
        loss = criterion(output, labels)
    
    # 缩放损失，防止 FP16 梯度下溢
    scaler.scale(loss).backward()
    
    # 梯度裁剪
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    scaler.step(optimizer)
    scaler.update()
```

---

## 分布式训练调试

### 常见问题

**1. 进程挂起（Hang）**
```bash
# 通常是某个进程的 HCCL 通信未完成
# 检查所有进程是否都到达了通信点
export HCCL_TIMEOUT=300  # 设置超时，超时后报错而非无限等待
```

**2. 梯度不一致**
```python
# 验证各进程梯度是否一致
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        dist.all_reduce(grad_norm)
        grad_norm /= dist.get_world_size()
        print(f"{name}: grad_norm={grad_norm:.4f}")
```

**3. 内存不均衡**
```python
# 监控各卡内存使用
for rank in range(world_size):
    if dist.get_rank() == rank:
        free, total = torch.npu.mem_get_info()
        print(f"Rank {rank}: {(total-free)/1024**3:.1f}GB used")
    dist.barrier()
```
