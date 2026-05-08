# CANN 生态全景概览

## 什么是 CANN

CANN（Compute Architecture for Neural Networks）是华为面向 AI 场景打造的**异构计算架构**，定位是昇腾 AI 处理器的软件使能平台，相当于昇腾生态中的"CUDA"。

CANN 于 2018 年随昇腾芯片一同发布，2025 年宣布全栈开源，标志着昇腾生态从封闭走向开放。

---

## CANN 在整个 AI 栈中的位置

```
┌─────────────────────────────────────────────────────┐
│              AI 应用 / 大模型 / 推理服务              │
├─────────────────────────────────────────────────────┤
│     MindSpore / PyTorch / TensorFlow / PaddlePaddle  │
├─────────────────────────────────────────────────────┤
│                  CANN 软件栈                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │AscendCL  │ │  GE图引擎 │ │  HCCL   │ │  AOE   │ │
│  │应用开发层 │ │ 编译执行层 │ │ 集合通信 │ │ 调优引擎│ │
│  └──────────┘ └──────────┘ └──────────┘ └────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │  TBE/    │ │  ATC     │ │ Runtime  │            │
│  │ Ascend C │ │ 模型转换  │ │  运行时  │            │
│  └──────────┘ └──────────┘ └──────────┘            │
├─────────────────────────────────────────────────────┤
│              昇腾驱动层（NPU Driver）                 │
├─────────────────────────────────────────────────────┤
│         昇腾 AI 处理器（达芬奇架构 NPU）              │
│    Ascend 310 / 910 / 910B / 910C / Atlas 系列       │
└─────────────────────────────────────────────────────┘
```

---

## CANN 核心组件一览

| 组件 | 全称 | 核心职责 |
|------|------|----------|
| **AscendCL** | Ascend Computing Language | 应用开发 C/C++ API，设备/内存/模型管理 |
| **GE** | Graph Engine | 图优化、图编译、图执行全流程 |
| **ATC** | Ascend Tensor Compiler | 离线模型转换工具（.pb/.onnx → .om） |
| **TBE** | Tensor Boost Engine | 算子开发框架（DSL/TIK 两种模式） |
| **Ascend C** | Ascend C | 新一代 C++ 算子编程语言 |
| **HCCL** | Huawei Collective Communication Library | 分布式训练集合通信库 |
| **AOE** | Ascend Optimization Engine | 算子/子图/梯度自动调优引擎 |
| **Runtime** | CANN Runtime | 硬件资源管理、任务调度、流管理 |
| **Driver** | NPU Driver | 内核态驱动，硬件抽象层 |
| **AOL** | Ascend Operator Library | 预置算子库（1500+ 基础算子） |
| **Framework Adaptor** | 框架适配器 | 对接 PyTorch/TF/MindSpore 等框架 |

---

## CANN 分层架构详解

CANN 自顶向下分为五大层级：

### 1. 计算语言层
以 **AscendCL** 为核心，提供面向应用开发者的 C/C++ API。开发者通过 AscendCL 完成：
- 设备初始化与上下文管理
- 内存分配与数据传输
- 模型加载与推理执行
- 媒体数据预处理（DVPP）

### 2. 计算服务层
包含算子加速库（AOL）、调优引擎（AOE）和框架适配器（Framework Adaptor）。这一层屏蔽了底层硬件差异，为上层框架提供统一的算子服务。

### 3. 计算编译层
核心是**图编译器**和 **TBE 张量加速引擎**。将框架产生的计算图（IR）编译为昇腾硬件可执行的二进制模型（.om 文件）。

### 4. 计算执行层
包含 **Runtime 运行时**、**GE 图引擎**和 **HCCL 集合通信库**。负责在运行时管理硬件资源、调度任务流、执行分布式通信。

### 5. 计算基础层
达芬奇架构硬件，包含 AI Core（矩阵/向量/标量计算单元）、AI CPU、DVPP、内存控制器等物理单元。

---

## 开发者视角：三条主要开发路径

```
路径一：应用推理开发
  框架训练 → ATC 模型转换 → AscendCL 加载推理 → 部署

路径二：自定义算子开发
  TBE DSL / Ascend C 编写算子 → 注册到框架 → 训练/推理使用

路径三：分布式训练开发
  框架 + HCCL → 多卡/多机并行训练 → AOE 调优
```

---

## 与 CUDA 生态的对比

| 维度 | NVIDIA CUDA | 华为 CANN |
|------|-------------|-----------|
| 硬件 | GPU（CUDA Core + Tensor Core） | NPU（达芬奇 AI Core） |
| 应用 API | CUDA Runtime API | AscendCL |
| 算子开发 | CUDA C / Triton | Ascend C / TBE |
| 模型转换 | TensorRT | ATC |
| 集合通信 | NCCL | HCCL |
| 调优工具 | Nsight | MindStudio Profiler |
| 框架支持 | PyTorch/TF/JAX | PyTorch/TF/MindSpore |

---

## 本知识体系的阅读路线

**入门路线**（推理应用开发）：
> 第1章硬件基础 → 第3章AscendCL → 第5章ATC模型转换 → 第9章部署

**进阶路线**（算子开发）：
> 第2章达芬奇架构 → 第4章算子开发 → 第5章图引擎 → 第8章调优

**高阶路线**（分布式训练）：
> 第6章HCCL → 第7章框架适配 → 第8章调优工具 → 第9章生态
