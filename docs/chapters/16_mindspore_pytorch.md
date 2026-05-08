# MindSpore 与 PyTorch 在昇腾上的开发实践

## MindSpore 概述

MindSpore 是华为自研的 AI 框架，与昇腾硬件深度协同优化，是 CANN 生态的原生框架。

**MindSpore 核心特性**：
- 原生支持昇腾硬件，无需额外适配插件
- 静态图（Graph Mode）和动态图（PyNative Mode）双模式
- 自动微分、自动并行
- 与 CANN 深度集成，性能最优

---

## MindSpore 基础使用

### 安装

```bash
# 安装 MindSpore（昇腾版本）
pip install mindspore==2.3.0

# 验证
python -c "import mindspore; mindspore.set_context(device_target='Ascend'); print('OK')"
```

### 基本张量操作

```python
import mindspore as ms
import mindspore.ops as ops
import numpy as np

# 设置运行设备
ms.set_context(device_target="Ascend", device_id=0)

# 创建张量
x = ms.Tensor(np.random.randn(3, 4), dtype=ms.float32)
y = ms.Tensor(np.random.randn(3, 4), dtype=ms.float32)

# 基本运算
z = x + y
w = ops.relu(x)
print(z.shape, z.dtype)
```

### 模型定义

```python
import mindspore.nn as nn
import mindspore.ops as ops

class ResidualBlock(nn.Cell):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, pad_mode='pad')
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    
    def construct(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)

class SimpleResNet(nn.Cell):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = ResidualBlock(64)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Dense(64, num_classes)
    
    def construct(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)
```

### 训练流程

```python
import mindspore as ms
import mindspore.nn as nn
from mindspore import Model, Tensor
from mindspore.train.callback import LossMonitor, ModelCheckpoint, CheckpointConfig

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

# 模型、损失函数、优化器
model_net = SimpleResNet(num_classes=10)
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.Adam(model_net.trainable_params(), learning_rate=1e-3)

# 封装为 Model
model = Model(model_net, loss_fn=loss_fn, optimizer=optimizer, metrics={"acc"})

# 回调配置
ckpt_config = CheckpointConfig(save_checkpoint_steps=100, keep_checkpoint_max=5)
ckpt_callback = ModelCheckpoint(prefix="resnet", directory="./checkpoints", config=ckpt_config)

# 训练
model.train(
    epoch=10,
    train_dataset=train_dataset,
    callbacks=[LossMonitor(10), ckpt_callback],
    dataset_sink_mode=True  # 数据下沉模式，性能更好
)

# 评估
result = model.eval(eval_dataset, dataset_sink_mode=True)
print(f"Accuracy: {result['acc']:.4f}")
```

---

## MindSpore 自动并行

MindSpore 提供强大的自动并行能力：

```python
import mindspore as ms
from mindspore.communication import init

# 初始化分布式通信
init()

# 设置并行模式
ms.set_auto_parallel_context(
    parallel_mode=ms.ParallelMode.AUTO_PARALLEL,  # 自动并行
    gradients_mean=True,
    full_batch=False
)

# 或手动设置并行策略
ms.set_auto_parallel_context(
    parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL,
    device_num=8
)

# 在模型中标注并行策略
class ParallelDense(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = ops.MatMul()
        # 设置算子并行策略：输入按 batch 维度切分，权重不切分
        self.matmul.shard(((8, 1), (1, 1)))
    
    def construct(self, x, w):
        return self.matmul(x, w)
```

---

## PyTorch 在昇腾上的最佳实践

### 性能优化技巧

```python
import torch
import torch_npu

# 1. 使用 channels_last 内存格式（卷积网络）
model = model.to(memory_format=torch.channels_last)
inputs = inputs.to(memory_format=torch.channels_last)

# 2. 开启 JIT 融合
torch.npu.set_option({"ACL_OP_COMPILER_CACHE_MODE": "enable"})

# 3. 使用 FP16 训练
model = model.half()
inputs = inputs.half()

# 4. 数据预取
class PrefetchDataLoader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.npu.Stream()
    
    def __iter__(self):
        first = True
        for next_data in self.dataloader:
            with torch.npu.stream(self.stream):
                next_data = [d.to(self.device, non_blocking=True) for d in next_data]
            if not first:
                yield current_data
            torch.npu.current_stream().wait_stream(self.stream)
            current_data = next_data
            first = False
        yield current_data
```

### 模型推理优化

```python
import torch
import torch_npu

# 推理模式（关闭梯度计算）
model.eval()
with torch.no_grad():
    with torch.autocast(device_type='npu', dtype=torch.float16):
        output = model(input)

# 使用 torch.compile 加速（PyTorch 2.x）
compiled_model = torch.compile(model, backend="npu", mode="reduce-overhead")

# 批量推理
def batch_inference(model, inputs, batch_size=32):
    results = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size].npu()
        with torch.no_grad():
            output = model(batch)
        results.append(output.cpu())
    return torch.cat(results)
```

---

## 框架选择建议

| 场景 | 推荐框架 | 原因 |
|------|---------|------|
| 新项目（华为生态） | MindSpore | 原生支持，性能最优 |
| 从 PyTorch 迁移 | torch_npu | 代码改动最小 |
| 推理部署 | AscendCL | 最轻量，性能最优 |
| 科研探索 | PyTorch + torch_npu | 生态最丰富 |
| 大模型训练 | MindSpore / torch_npu + DeepSpeed | 自动并行支持好 |

---

## 代码迁移：CUDA → 昇腾

### PyTorch 代码迁移

```python
# CUDA 代码
device = torch.device("cuda:0")
x = x.cuda()
model = model.cuda()
torch.cuda.synchronize()

# 昇腾代码（最小改动）
import torch_npu
device = torch.device("npu:0")
x = x.npu()
model = model.npu()
torch.npu.synchronize()
```

### 自动迁移工具

```bash
# 使用 MindStudio 的代码迁移工具
# 或使用命令行工具
ms-migration --input=cuda_code.py --output=npu_code.py --framework=pytorch
```

迁移工具会自动替换：
- `cuda` → `npu`
- `torch.cuda.*` → `torch.npu.*`
- CUDA 特有 API → 昇腾等价 API
