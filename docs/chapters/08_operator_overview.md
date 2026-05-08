# 算子开发体系概览

## 什么是算子

在深度学习框架中，**算子（Operator）** 是计算图的基本执行单元，对应一种数学运算（如矩阵乘法、卷积、激活函数等）。

在昇腾平台上，算子分为两类：

| 类型 | 执行单元 | 适用场景 |
|------|---------|----------|
| **AI Core 算子** | 达芬奇 AI Core | 矩阵/向量密集计算（卷积、GEMM、激活等） |
| **AI CPU 算子** | AI CPU（ARM） | 控制流复杂、数据依赖强的算子（TopK、Sort等） |

---

## 算子开发技术路线

CANN 提供了多种算子开发方式，从高层到底层：

```
┌─────────────────────────────────────────────────────────┐
│  层级        技术          特点                          │
├─────────────────────────────────────────────────────────┤
│  最高层   TBE DSL         声明式，自动调度，开发最简单    │
│           (Python)        适合规则算子                   │
├─────────────────────────────────────────────────────────┤
│  中间层   TBE TIK         命令式，手动控制数据流          │
│           (Python)        适合复杂算子，性能可控          │
├─────────────────────────────────────────────────────────┤
│  新一代   Ascend C        C++ 语法，最接近硬件            │
│           (C++)           官方推荐，替代 TBE TIK          │
├─────────────────────────────────────────────────────────┤
│  最底层   汇编/指令集      极致性能，开发难度极高          │
│           (极少使用)       通常由华为内部使用              │
└─────────────────────────────────────────────────────────┘
```

---

## 算子开发流程

无论使用哪种技术，算子开发的基本流程一致：

```
1. 算子分析
   ├── 确定算子功能（数学定义）
   ├── 分析输入输出规格（shape/dtype/format）
   └── 评估计算访存比，选择开发方式

2. 算子实现
   ├── 编写 Kernel 函数（Ascend C / TBE）
   ├── 实现 Tiling 策略（分块计算）
   └── 处理边界条件

3. 算子信息定义
   ├── 编写算子原型（OpProto）
   ├── 定义支持的数据类型和格式
   └── 注册算子信息

4. 算子编译
   ├── 使用 ATC 或 acl_op_compiler 编译
   └── 生成 .o 二进制文件

5. 算子验证
   ├── 精度验证（与 CPU 参考实现对比）
   ├── 性能测试
   └── 边界条件测试

6. 算子注册
   ├── 注册到 PyTorch/MindSpore 等框架
   └── 集成到训练/推理流程
```

---

## 算子库（AOL）

CANN 内置了丰富的预置算子库（AOL，Ascend Operator Library）：

### 基础算子分类

| 类别 | 典型算子 |
|------|---------|
| 矩阵运算 | MatMul, BatchMatMul, GEMM |
| 卷积 | Conv2D, DepthwiseConv2D, Conv3D |
| 归一化 | BatchNorm, LayerNorm, GroupNorm |
| 激活函数 | ReLU, GELU, Sigmoid, Tanh, SiLU |
| 池化 | MaxPool, AvgPool, GlobalAvgPool |
| 注意力 | FlashAttention, SelfAttention |
| 归约 | ReduceSum, ReduceMean, ReduceMax |
| 元素级 | Add, Mul, Sub, Div, Pow |
| 形状变换 | Reshape, Transpose, Concat, Split |
| 嵌入 | Embedding, GatherV2 |

### 融合算子

融合算子将多个基础算子合并为一个，减少内存读写次数：

```
LayerNorm + GELU → LayerNormGelu（融合算子）
减少中间结果写回 HBM，提升 35%+ 吞吐量
```

常见融合算子：
- `LayerNormGelu`：LayerNorm + GELU
- `AddLayerNorm`：Add + LayerNorm
- `MatMulBiasAdd`：MatMul + BiasAdd
- `FlashAttentionScore`：完整注意力计算

---

## 算子调试工具

### CPU 侧仿真（Ascend C）

Ascend C 支持在 CPU 上仿真执行，无需真实硬件即可调试：

```bash
# 编译为 CPU 仿真版本
cmake .. -DASCEND_COMPUTE_UNIT=cpu

# 运行仿真
./operator_test
```

### 精度比对工具

```python
# 使用 msaccucmp 工具比对精度
# 比较 NPU 输出与 CPU 参考输出的差异
from msaccucmp import compare_dump_files

compare_dump_files(
    npu_dump_path="./npu_dump",
    cpu_dump_path="./cpu_dump",
    output_path="./compare_result"
)
```

### 算子性能分析

```bash
# 使用 Profiling 工具分析算子耗时
msprof --application=./inference_app \
       --output=./profiling_output \
       --ai-core-metrics=pipe_utilization
```

---

## 算子注册机制

### 注册到 PyTorch

```python
# 方式一：使用 torch_npu 扩展
import torch
import torch_npu

# 自定义算子注册
torch.ops.load_library("libcustom_op.so")

# 调用自定义算子
result = torch.ops.custom_op.my_operator(input_tensor)
```

### 注册到 MindSpore

```python
# 使用 MindSpore 自定义算子接口
from mindspore.ops import CustomRegOp, custom_info_register

custom_op_info = CustomRegOp("MyOp") \
    .input(0, "x") \
    .output(0, "y") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default) \
    .get_op_info()

@custom_info_register(custom_op_info)
def my_op_impl(x):
    # 算子实现
    pass
```

---

## 算子开发最佳实践

### 1. 优先使用内置算子

在开发自定义算子前，先检查 AOL 是否已有满足需求的算子。内置算子经过充分优化，性能通常优于自定义实现。

### 2. 合理设计 Tiling

Tiling 策略直接影响性能：
- Tile 太小：计算效率低，调度开销大
- Tile 太大：超出 L1/UB 容量，导致溢出

```
最优 Tile 大小 ≈ L1 Buffer 容量 / (输入 + 输出数据大小)
```

### 3. 利用双缓冲

```
Buffer A: 计算中
Buffer B: 预取下一块数据
→ 交替使用，隐藏内存延迟
```

### 4. 数据类型选择

- 训练：优先 FP16/BF16，精度敏感处用 FP32
- 推理：优先 INT8（量化），精度要求高时用 FP16

### 5. 使用 AOE 自动调优

```bash
# 使用 AOE 自动搜索最优 Tiling 参数
aoe --framework=5 \
    --model=model.onnx \
    --job_type=2 \
    --output=./aoe_result
```
