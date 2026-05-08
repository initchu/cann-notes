# Framework Adaptor：框架适配层

## Framework Adaptor 概述

Framework Adaptor 是 CANN 软件栈中连接上层 AI 框架与底层 CANN 运行时的适配层。它将框架的计算图和算子调用转换为 CANN 的内部表示，使 PyTorch、TensorFlow、MindSpore 等框架能够无缝运行在昇腾硬件上。

```
PyTorch / TensorFlow / MindSpore
    ↓ 框架原生 API
Framework Adaptor
    ├── 算子映射（框架算子 → 昇腾算子）
    ├── 数据格式转换（NCHW ↔ NC1HWC0）
    ├── 内存管理适配
    └── 图优化接入
    ↓
CANN GE / Runtime
    ↓
昇腾 AI 处理器
```

---

## PyTorch 适配：torch_npu

`torch_npu` 是华为为 PyTorch 开发的昇腾适配插件，是使用 PyTorch 在昇腾上训练/推理的核心组件。

### 安装

```bash
# 安装 torch_npu（版本需与 PyTorch 和 CANN 版本匹配）
pip install torch_npu==2.1.0

# 验证安装
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
```

### 基本使用

```python
import torch
import torch_npu

# 检查 NPU 可用性
print(torch.npu.is_available())      # True
print(torch.npu.device_count())      # 8（8卡机器）
print(torch.npu.current_device())    # 0

# 张量操作
x = torch.randn(3, 4).npu()         # 创建 NPU 张量
y = torch.randn(3, 4).npu(0)        # 指定设备 0
z = x + y                            # NPU 上计算

# 设备间移动
x_cpu = x.cpu()                      # NPU → CPU
x_npu = x_cpu.npu()                  # CPU → NPU

# 指定设备
with torch.npu.device(1):
    x = torch.randn(3, 4).npu()     # 在设备 1 上创建
```

### 模型训练

```python
import torch
import torch_npu
import torch.nn as nn

# 模型定义
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# 移到 NPU
device = torch.device("npu:0")
model = SimpleModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(10):
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### 自动混合精度

```python
from torch.npu.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, labels in dataloader:
    inputs, labels = inputs.npu(), labels.npu()
    
    with autocast():  # 自动使用 FP16
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 算子映射机制

Framework Adaptor 维护了一张算子映射表，将框架算子映射到昇腾算子：

```python
# torch_npu 内部的算子注册示例（概念性）
@torch.library.impl("aten::relu", "PrivateUse1")
def npu_relu(self):
    return torch_npu._C._VariableFunctions.npu_relu(self)

# 自定义算子注册
torch.library.define("mylib::custom_op", "(Tensor x) -> Tensor")

@torch.library.impl("mylib::custom_op", "PrivateUse1")
def custom_op_npu(x):
    return torch_npu._C._VariableFunctions.custom_op_impl(x)
```

### 查看算子支持情况

```python
import torch_npu

# 查看当前支持的算子列表
ops = torch_npu.get_npu_supported_ops()
print(f"Supported ops: {len(ops)}")

# 检查特定算子是否支持
print("relu" in ops)
```

---

## 数据格式转换

昇腾内部使用优化的数据格式（如 NC1HWC0），Framework Adaptor 自动处理格式转换：

```python
# 用户代码使用标准 NCHW 格式
x = torch.randn(1, 64, 56, 56).npu()  # NCHW

# Framework Adaptor 内部自动转换为 NC1HWC0
# NC1HWC0: N=1, C1=4(64/16), H=56, W=56, C0=16
# 对用户透明，无需手动处理

# 卷积操作（内部使用 NC1HWC0 格式）
conv = nn.Conv2d(64, 128, 3, padding=1).npu()
y = conv(x)  # 输出仍然是 NCHW 视图
```

---

## 图模式加速

torch_npu 支持图模式，将动态图转换为静态图执行：

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu

# 方式一：使用 torch.compile（PyTorch 2.x）
model = torch.compile(model, backend="npu")

# 方式二：使用 torch.jit.script
scripted_model = torch.jit.script(model)
scripted_model = scripted_model.npu()

# 方式三：使用 torch.jit.trace
example_input = torch.randn(1, 3, 224, 224).npu()
traced_model = torch.jit.trace(model, example_input)
```

---

## TensorFlow 适配

```python
# 安装 TF 昇腾插件
pip install tensorflow-cpu==2.6.5
pip install npu-bridge

import tensorflow as tf
import npu_bridge

# 配置 NPU
from npu_bridge.npu_init import *
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True

# 使用 NPU 执行
with tf.Session(config=config) as sess:
    # 正常的 TF 代码，自动在 NPU 上执行
    result = sess.run(output_tensor, feed_dict={input_tensor: data})
```

---

## 常见适配问题

### 1. 算子不支持

```python
# 错误：某算子在 NPU 上不支持
# RuntimeError: "xxx" not implemented for 'PrivateUse1'

# 解决方案一：使用等价算子
# 解决方案二：将该算子移到 CPU 执行
x_cpu = x.cpu()
result_cpu = unsupported_op(x_cpu)
result_npu = result_cpu.npu()

# 解决方案三：开发自定义算子
```

### 2. 数据类型不支持

```python
# 某些算子不支持 FP64
x = x.float()  # FP64 → FP32

# 或使用精度模式配置
torch.npu.set_option({"ACL_PRECISION_MODE": "allow_fp32_to_fp16"})
```

### 3. 内存格式问题

```python
# 确保张量是连续的
x = x.contiguous()

# 或使用 channels_last 格式（某些场景更高效）
x = x.to(memory_format=torch.channels_last)
model = model.to(memory_format=torch.channels_last)
```

### 4. 随机数一致性

```python
# 设置随机种子确保可复现
torch.manual_seed(42)
torch.npu.manual_seed(42)
torch.npu.manual_seed_all(42)
```
