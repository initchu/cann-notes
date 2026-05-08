# GE 图引擎：图优化与执行

## GE 概述

GE（Graph Engine）是 CANN 的核心图处理引擎，负责接收来自 AI 框架的计算图，执行一系列优化，并将其编译为昇腾硬件可执行的格式。

```
AI 框架（PyTorch/MindSpore/TF）
    ↓ 导出计算图（IR）
GE 图引擎
    ├── 图优化（Graph Optimization）
    ├── 图编译（Graph Compilation）
    └── 图执行（Graph Execution）
    ↓
昇腾 AI 处理器执行
```

---

## GE 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                      GE 处理流程                             │
│                                                             │
│  输入图（IR）                                               │
│      ↓                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  图准备阶段（Graph Prepare）                          │   │
│  │  • 图合法性检查                                       │   │
│  │  • 算子类型推导                                       │   │
│  │  • 形状推导（Shape Inference）                        │   │
│  └─────────────────────────────────────────────────────┘   │
│      ↓                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  图优化阶段（Graph Optimize）                         │   │
│  │  • 常量折叠（Constant Folding）                       │   │
│  │  • 算子融合（Operator Fusion）                        │   │
│  │  • 内存复用（Memory Reuse）                           │   │
│  │  • 数据格式转换优化                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│      ↓                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  图分割阶段（Graph Partition）                        │   │
│  │  • AI Core 子图 vs AI CPU 子图分割                    │   │
│  │  • 跨设备子图分割（多卡场景）                          │   │
│  └─────────────────────────────────────────────────────┘   │
│      ↓                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  图编译阶段（Graph Build）                            │   │
│  │  • 算子编译（TBE/Ascend C → 二进制）                  │   │
│  │  • 内存规划（Memory Planning）                        │   │
│  │  • 任务生成（Task Generation）                        │   │
│  └─────────────────────────────────────────────────────┘   │
│      ↓                                                      │
│  可执行模型（.om 文件）                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 图优化技术详解

### 1. 常量折叠（Constant Folding）

将编译期可确定的常量计算提前执行，减少运行时计算量：

```
优化前：
  x = input
  c1 = Const(2.0)
  c2 = Const(3.0)
  c3 = Add(c1, c2)    ← 常量加法
  y = Mul(x, c3)

优化后：
  x = input
  c3 = Const(5.0)     ← 编译期计算完成
  y = Mul(x, c3)
```

### 2. 算子融合（Operator Fusion）

将多个算子合并为一个，减少内存读写次数：

```
优化前：
  x → Conv2D → BN → ReLU → y
  （3次写 HBM + 3次读 HBM）

优化后：
  x → Conv2D_BN_ReLU → y
  （1次写 HBM + 1次读 HBM）
  
性能提升：减少 ~67% 的内存带宽消耗
```

**GE 支持的融合模式**：

| 融合类型 | 示例 |
|----------|------|
| 垂直融合 | Conv + BN + ReLU |
| 水平融合 | 多个相同算子并行 |
| 子图融合 | 复杂子图替换为单算子 |

### 3. 内存复用（Memory Reuse）

分析张量的生命周期，让不同张量共享内存：

```
优化前：
  tensor_a: 分配 100MB，生命周期 [step1, step3]
  tensor_b: 分配 100MB，生命周期 [step4, step6]
  总内存：200MB

优化后：
  tensor_a 和 tensor_b 共享同一块 100MB 内存
  总内存：100MB（节省 50%）
```

### 4. 数据格式优化

自动插入格式转换算子，使数据以最优格式流转：

```
框架输入：NCHW 格式
    ↓ GE 自动插入 TransData 算子
AI Core 计算：NC1HWC0 格式（5D，C 维度 16 对齐）
    ↓ GE 自动插入 TransData 算子
框架输出：NCHW 格式
```

---

## 图中间表示（IR）

GE 使用自定义的图 IR 格式，基于 Protobuf 定义：

```protobuf
// 简化的图 IR 结构
message GraphDef {
    string name = 1;
    repeated NodeDef node = 2;
}

message NodeDef {
    string name = 1;
    string op = 2;           // 算子类型
    repeated string input = 3; // 输入节点名
    map<string, AttrValue> attr = 4; // 算子属性
}
```

### 查看图结构

```python
# 使用 MindStudio 可视化图结构
# 或使用命令行工具
atc --model=model.pb \
    --framework=3 \
    --output=model \
    --soc_version=Ascend910B3 \
    --save_original_model=true  # 保存原始图用于分析
```

---

## 动态图与静态图

### 静态图（Static Graph）

- 编译期确定所有形状，运行时无需重新编译
- 性能最优
- 适合推理部署

```python
# MindSpore 静态图模式
import mindspore as ms
ms.set_context(mode=ms.GRAPH_MODE)
```

### 动态图（Dynamic Graph）

- 支持动态形状输入
- 运行时可能触发重新编译（JIT）
- 适合训练阶段

```python
# MindSpore 动态图模式
ms.set_context(mode=ms.PYNATIVE_MODE)
```

### 动态形状支持

GE 通过以下机制支持动态形状：

1. **档位编译**：预编译多个固定形状版本，运行时选择最接近的
2. **动态编译**：运行时根据实际形状 JIT 编译
3. **符号形状**：使用符号表示动态维度，编译一次支持多种形状

---

## GE 配置与调优

### 通过 AscendCL 配置 GE

```c
// 设置 GE 图优化级别
aclSetCompileopt(ACL_OP_COMPILE_DEFAULT, "enable_scope_fusion_passes=true");

// 设置精度模式
aclSetCompileopt(ACL_PRECISION_MODE, "allow_fp32_to_fp16");

// 设置算子编译缓存
aclSetCompileopt(ACL_OP_COMPILER_CACHE_MODE, "enable");
aclSetCompileopt(ACL_OP_COMPILER_CACHE_DIR, "./op_cache");
```

### 通过环境变量配置

```bash
# 开启图优化日志
export ASCEND_GLOBAL_LOG_LEVEL=1

# 设置算子编译缓存目录
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp

# 开启 GE 图 dump（调试用）
export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=3
export DUMP_GRAPH_PATH=./ge_graph_dump
```

---

## 图执行模式

### 同步执行

```c
// 同步执行：等待图执行完成
aclmdlExecute(modelId, inputDataset, outputDataset);
```

### 异步执行

```c
// 异步执行：提交到 Stream，立即返回
aclmdlExecuteAsync(modelId, inputDataset, outputDataset, stream);

// 等待完成
aclrtSynchronizeStream(stream);
```

### 多图并发执行

```c
// 在不同 Stream 上并发执行多个图
aclmdlExecuteAsync(model1Id, input1, output1, stream1);
aclmdlExecuteAsync(model2Id, input2, output2, stream2);

// 等待所有图完成
aclrtSynchronizeStream(stream1);
aclrtSynchronizeStream(stream2);
```

---

## 常见问题排查

### 图编译失败

```bash
# 查看详细错误信息
export ASCEND_GLOBAL_LOG_LEVEL=0  # DEBUG 级别

# 常见原因：
# 1. 算子不支持当前数据类型/格式
# 2. 输入形状不满足算子约束
# 3. 算子版本与 CANN 版本不匹配
```

### 图执行结果不正确

```bash
# 开启算子 Dump，对比每个算子的输入输出
export ENABLE_DUMP=1
export DUMP_PATH=./dump_data
export DUMP_STEP=0

# 使用 msaccucmp 工具比对精度
python msaccucmp.py compare -m ./dump_data -g ./golden_data
```
