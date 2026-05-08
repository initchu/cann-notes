# 昇腾 AI 处理器硬件体系

## 昇腾芯片家族

华为昇腾（Ascend）系列 AI 处理器是 CANN 生态的硬件基础。按照应用场景分为推理芯片和训练芯片两大系列。

### 主要芯片型号

| 芯片型号 | 定位 | 算力（FP16） | 典型产品 |
|----------|------|-------------|----------|
| Ascend 310 | 边缘推理 | 16 TOPS | Atlas 200 DK |
| Ascend 310P | 边缘推理增强 | 32 TOPS | Atlas 300I Pro |
| Ascend 910 | 云端训练 | 256 TFLOPS | Atlas 800 训练服务器 |
| Ascend 910B | 云端训练增强 | 320 TFLOPS | Atlas 800T A2 |
| Ascend 910C | 最新旗舰 | ~600 TFLOPS | Atlas 900 超节点 |

### Atlas 产品系列

```
Atlas 200 系列    ── 边缘推理，开发者套件
Atlas 300 系列    ── PCIe 推理加速卡
Atlas 500 系列    ── 智能小站，边缘计算
Atlas 800 系列    ── 数据中心训练/推理服务器
Atlas 900 系列    ── 超大规模 AI 集群
Atlas A2 系列     ── 新一代训练平台
```

---

## 昇腾处理器逻辑架构

昇腾 AI 处理器的逻辑架构由四大模块组成：

```
┌─────────────────────────────────────────────────────────┐
│                   昇腾 AI 处理器                         │
│                                                         │
│  ┌─────────────┐    ┌──────────────────────────────┐   │
│  │  Control CPU │    │       AI 计算引擎             │   │
│  │  (ARM/x86)  │    │  ┌──────────┐  ┌──────────┐  │   │
│  │             │    │  │  AI Core │  │  AI CPU  │  │   │
│  │  任务调度    │    │  │ (矩阵计算)│  │(标量计算) │  │   │
│  │  资源管理    │    │  └──────────┘  └──────────┘  │   │
│  └─────────────┘    └──────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              多级缓存体系                         │   │
│  │   L1 Buffer → L2 Cache → HBM (高带宽内存)        │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────┐    ┌──────────────────────────────┐  │
│  │     DVPP     │    │       互联接口                │  │
│  │ 数字视觉预处理│    │  PCIe / HCCS / RoCE          │  │
│  └──────────────┘    └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## AI Core：核心计算单元

AI Core 是昇腾处理器的核心，基于**达芬奇架构**设计，内部包含三类计算单元：

### 矩阵计算单元（Cube Unit）
- 专为矩阵乘法（GEMM）设计
- 支持 FP16、INT8、BF16 等数据类型
- 单个 AI Core 每个时钟周期可完成 4096 次 FP16 乘加运算
- 对应 CUDA 中的 Tensor Core

### 向量计算单元（Vector Unit）
- 执行逐元素运算（激活函数、归一化等）
- 支持 FP32/FP16/INT32 等多种精度
- 对应 CUDA 中的 CUDA Core（向量模式）

### 标量计算单元（Scalar Unit）
- 处理控制流、地址计算、循环逻辑
- 类似 CPU 的通用计算能力

### AI Core 内部存储层次

```
寄存器文件（Register File）
    ↕ 极低延迟
本地内存（Local Memory / UB - Unified Buffer）
    ↕ 低延迟，高带宽
L1 缓存（L1 Buffer）
    ↕
L2 缓存（L2 Cache）
    ↕
HBM（High Bandwidth Memory）
    ↕ 高带宽，较高延迟
DDR / 主机内存
```

---

## AI CPU

AI CPU 是昇腾处理器中的通用计算单元，基于 ARM 架构：

- 负责执行无法在 AI Core 上高效运行的算子（如 TopK、Sort 等）
- 处理控制流密集型任务
- 与 AI Core 协同工作，形成异构计算

---

## DVPP：数字视觉预处理

DVPP（Digital Vision Pre-Processing）是昇腾处理器内置的硬件图像/视频处理单元：

| 功能模块 | 说明 |
|----------|------|
| JPEGD | JPEG 图像解码 |
| JPEGE | JPEG 图像编码 |
| PNGD | PNG 图像解码 |
| VDEC | 视频解码（H.264/H.265） |
| VENC | 视频编码 |
| VPC | 视觉预处理（缩放/裁剪/色彩转换） |

DVPP 的优势在于将图像预处理从 CPU 卸载到专用硬件，大幅降低推理流水线的 CPU 瓶颈。

---

## 内存体系

### HBM（High Bandwidth Memory）
- 昇腾 910 系列配备 HBM2/HBM2e
- 带宽可达 900 GB/s 以上
- 容量：32GB（910）/ 64GB（910B）

### 内存访问模式

```
Host（CPU 侧）内存
    ↕ PCIe / HCCS 互联
Device（NPU 侧）HBM
    ↕ 内部总线
AI Core 本地缓存（L1/L2/UB）
```

开发者需要显式管理 Host 与 Device 之间的数据传输（类似 CUDA 的 cudaMemcpy）。

---

## 多卡互联：HCCS

HCCS（Huawei Cache Coherence System）是昇腾处理器间的高速互联总线：

- 单机 8 卡通过 HCCS 全互联，带宽远高于 PCIe
- 支持 Cache 一致性，简化多卡编程
- 类比 NVIDIA 的 NVLink

### Atlas 900 集群互联

```
单节点（8 × Ascend 910）
    ↕ HCCS（节点内）
多节点（通过 RoCE 网络）
    ↕ 100GbE / 400GbE RoCE
超节点（Atlas 900）
```

---

## 关键性能指标理解

### TOPS vs TFLOPS
- **TOPS**（Tera Operations Per Second）：整数/定点运算，常用于推理芯片
- **TFLOPS**（Tera Floating Point Operations Per Second）：浮点运算，常用于训练芯片

### 有效算力 vs 峰值算力
峰值算力是理论上限，实际应用中受限于：
- 内存带宽（Memory Bandwidth Bound）
- 计算访存比（Arithmetic Intensity）
- 算子融合程度
- 数据对齐要求

理解这些约束是进行 CANN 性能调优的基础。
