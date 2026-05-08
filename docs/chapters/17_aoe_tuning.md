# AOE 调优引擎

## AOE 概述

AOE（Ascend Optimization Engine）是 CANN 提供的自动化性能调优引擎，通过搜索最优的算子执行参数（如 Tiling 策略）来提升模型性能，无需开发者手动调优。

**AOE 支持三种调优模式**：

| 模式 | 全称 | 调优对象 | 典型收益 |
|------|------|---------|---------|
| OPAT | Operator Auto Tuning | 单算子 Tiling 参数 | 10-30% |
| SGAT | Subgraph Auto Tuning | 子图级融合与调度 | 15-40% |
| GDAT | Gradient Auto Tuning | 梯度计算优化 | 5-20% |

---

## AOE 工作原理

```
┌─────────────────────────────────────────────────────────┐
│                    AOE 调优流程                          │
│                                                         │
│  1. 收集算子信息                                         │
│     ├── 算子类型、输入输出形状、数据类型                   │
│     └── 当前执行性能基线                                  │
│                                                         │
│  2. 搜索空间构建                                         │
│     ├── Tiling 参数空间（分块大小、循环顺序等）            │
│     └── 融合策略空间                                     │
│                                                         │
│  3. 自动搜索（贝叶斯优化 / 遗传算法）                     │
│     ├── 在真实硬件上执行候选配置                          │
│     └── 记录性能数据                                     │
│                                                         │
│  4. 生成调优知识库                                       │
│     └── 保存最优配置到 .json 文件                        │
│                                                         │
│  5. 推理/训练时加载调优结果                               │
│     └── 自动使用最优配置执行                              │
└─────────────────────────────────────────────────────────┘
```

---

## OPAT：算子自动调优

### 通过 ATC 触发 OPAT

```bash
# 在模型转换时同时进行算子调优
atc --model=model.onnx \
    --framework=5 \
    --output=model_tuned \
    --soc_version=Ascend910B3 \
    --input_shape="input:1,3,224,224" \
    --enable_scope_fusion_passes=true \
    --op_compiler_cache_mode=enable \
    --op_compiler_cache_dir=./op_cache \
    --aoe_mode=1  # 1=OPAT
```

### 独立运行 AOE 工具

```bash
# 方式一：基于模型文件调优
aoe --framework=5 \
    --model=model.onnx \
    --job_type=1 \
    --output=./aoe_result \
    --soc_version=Ascend910B3 \
    --input_shape="input:1,3,224,224"

# 方式二：基于已有 .om 文件调优
aoe --om=model.om \
    --job_type=1 \
    --output=./aoe_result \
    --soc_version=Ascend910B3
```

### 在 Python 中触发 OPAT

```python
import torch
import torch_npu

# 开启 AOE 调优模式
torch.npu.set_option({
    "ACL_OP_COMPILER_CACHE_MODE": "enable",
    "ACL_OP_COMPILER_CACHE_DIR": "./op_cache"
})

# 运行几个 step 收集算子信息
for i, batch in enumerate(dataloader):
    if i >= 10:  # 收集 10 个 step 的数据
        break
    output = model(batch)

# 触发 AOE 调优
torch_npu.npu.set_aoe("./aoe_result")
```

---

## SGAT：子图自动调优

SGAT 在子图级别进行优化，寻找最优的算子融合策略。

```bash
# 子图调优
aoe --framework=5 \
    --model=model.onnx \
    --job_type=2 \
    --output=./sgat_result \
    --soc_version=Ascend910B3 \
    --input_shape="input:1,3,224,224" \
    --tuning_time_limit=3600  # 调优时间限制（秒）
```

### 子图调优配置

```json
// sgat_config.json
{
    "job_type": "subgraph",
    "tuning_time_limit": 3600,
    "max_tuning_count": 100,
    "search_algorithm": "bayesian",
    "target_metric": "throughput"
}
```

---

## GDAT：梯度自动调优

GDAT 专门针对训练场景的梯度计算进行优化。

```bash
# 梯度调优（需要训练数据）
aoe --framework=5 \
    --model=model.onnx \
    --job_type=3 \
    --output=./gdat_result \
    --soc_version=Ascend910B3 \
    --input_shape="input:1,3,224,224;label:1" \
    --loss_name="loss"
```

---

## 加载调优结果

### ATC 转换时加载

```bash
atc --model=model.onnx \
    --framework=5 \
    --output=model_optimized \
    --soc_version=Ascend910B3 \
    --input_shape="input:1,3,224,224" \
    --op_compiler_cache_mode=enable \
    --op_compiler_cache_dir=./aoe_result  # 指向调优结果目录
```

### 运行时加载

```python
import torch_npu

# 加载 AOE 调优结果
torch.npu.set_option({
    "ACL_OP_COMPILER_CACHE_MODE": "enable",
    "ACL_OP_COMPILER_CACHE_DIR": "./aoe_result"
})
```

---

## 调优结果分析

### 调优报告

AOE 会生成详细的调优报告：

```
aoe_result/
├── summary.json          # 调优摘要
├── op_tuning/
│   ├── MatMul_fp16.json  # MatMul 算子调优结果
│   ├── Conv2D_fp16.json  # Conv2D 算子调优结果
│   └── ...
└── profiling/
    ├── before_tuning/    # 调优前性能数据
    └── after_tuning/     # 调优后性能数据
```

`summary.json` 示例：
```json
{
    "total_ops": 156,
    "tuned_ops": 89,
    "performance_improvement": "23.5%",
    "tuning_time": "1823s",
    "top_improved_ops": [
        {"op": "MatMul_1", "improvement": "45.2%"},
        {"op": "Conv2D_3", "improvement": "31.8%"},
        {"op": "LayerNorm_2", "improvement": "28.4%"}
    ]
}
```

---

## 调优最佳实践

### 1. 选择代表性输入

```bash
# 使用实际业务中最常见的输入形状进行调优
# 避免使用极端形状（太小或太大）
--input_shape="input:8,3,224,224"  # 使用实际 batch size
```

### 2. 分阶段调优

```bash
# 第一阶段：快速 OPAT（1-2小时）
aoe --job_type=1 --tuning_time_limit=3600

# 第二阶段：深度 SGAT（4-8小时）
aoe --job_type=2 --tuning_time_limit=28800
```

### 3. 调优结果复用

```bash
# 相同模型结构、相同硬件的调优结果可以复用
# 将调优结果纳入版本管理
git add aoe_result/
git commit -m "Add AOE tuning results for ResNet50 on 910B3"
```

### 4. 增量调优

```bash
# 模型更新后，只对变化的算子重新调优
aoe --model=model_v2.onnx \
    --job_type=1 \
    --output=./aoe_result_v2 \
    --base_result=./aoe_result  # 基于已有结果增量调优
```

---

## 性能调优效果示例

| 模型 | 调优前（ms） | 调优后（ms） | 提升 |
|------|------------|------------|------|
| ResNet50 | 3.2 | 2.4 | 25% |
| BERT-Base | 12.5 | 9.8 | 22% |
| YOLOv5s | 8.1 | 6.2 | 24% |
| GPT-2 | 45.3 | 35.1 | 22% |

*以上数据为示意，实际效果因硬件和模型而异*
