# 性能分析与调试工具

## 性能分析工具体系

CANN 提供了完整的性能分析工具链，帮助开发者定位性能瓶颈：

```
┌─────────────────────────────────────────────────────────┐
│                  性能分析工具体系                         │
│                                                         │
│  npu-smi          ── 实时硬件监控（类 nvidia-smi）        │
│  msprof           ── 命令行 Profiling 工具               │
│  MindStudio       ── 集成 IDE + 可视化分析               │
│  Profiling API    ── 代码内嵌 Profiling                  │
│  msaccucmp        ── 精度比对工具                        │
└─────────────────────────────────────────────────────────┘
```

---

## npu-smi：实时监控

```bash
# 基本状态查看
npu-smi info

# 持续监控（每秒刷新）
watch -n 1 npu-smi info

# 查看详细信息
npu-smi info -t common -i 0    # 设备 0 通用信息
npu-smi info -t usages -i 0    # 利用率
npu-smi info -t memory -i 0    # 内存使用
npu-smi info -t temp -i 0      # 温度
npu-smi info -t power -i 0     # 功耗
npu-smi info -t proc -i 0      # 进程信息

# 输出示例
# +-----------------------------------------------------------------------------------+
# | NPU  Name    Health  Power(W)  Temp(C)  AICore(%)  Memory-Usage(MB)              |
# +===================================================================================+
# | 0    910B3   OK      285.0     52       87         45312 / 65536                 |
# +-----------------------------------------------------------------------------------+
```

---

## msprof：命令行 Profiling

### 基本用法

```bash
# 对应用程序进行 Profiling
msprof --application="./inference_app arg1 arg2" \
       --output=./profiling_output \
       --ai-core-metrics=pipe_utilization

# 对 Python 脚本进行 Profiling
msprof --application="python train.py" \
       --output=./profiling_output \
       --ai-core-metrics=arithmetic_utilization
```

### 采集指标选项

```bash
# AI Core 指标
--ai-core-metrics=pipe_utilization        # 流水线利用率
--ai-core-metrics=arithmetic_utilization  # 算术单元利用率
--ai-core-metrics=memory_bandwidth        # 内存带宽
--ai-core-metrics=l2_cache               # L2 缓存命中率

# 采集范围
--task-time=on          # 任务执行时间
--aicpu=on              # AI CPU 算子
--hccl=on               # 通信算子
--runtime-api=on        # Runtime API 调用
--dvpp=on               # DVPP 操作
```

### 分析 Profiling 结果

```bash
# 解析 Profiling 数据
msprof --export=summary \
       --output=./profiling_output

# 生成的文件
profiling_output/
├── PROF_000001_20260416_100000/
│   ├── device_0/
│   │   ├── timeline/
│   │   │   └── msprof_*.json    # 时间线数据（可在 Chrome Tracing 查看）
│   │   ├── summary/
│   │   │   ├── op_summary.csv   # 算子耗时汇总
│   │   │   └── step_trace.csv   # 训练步骤追踪
│   │   └── aicore_intermediate/ # AI Core 详细数据
│   └── host/
│       └── runtime_api.csv      # Runtime API 调用记录
```

### 读取 op_summary.csv

```python
import pandas as pd

df = pd.read_csv("profiling_output/.../op_summary.csv")

# 按耗时排序，找出最慢的算子
top_ops = df.sort_values("Total Time(us)", ascending=False).head(20)
print(top_ops[["Op Name", "Op Type", "Total Time(us)", "Count"]])

# 计算各算子类型的总耗时占比
type_time = df.groupby("Op Type")["Total Time(us)"].sum()
print(type_time.sort_values(ascending=False))
```

---

## 代码内嵌 Profiling API

### Python API

```python
import torch
import torch_npu
from torch_npu.profiler import profile, ProfilerActivity

# 方式一：上下文管理器
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        output = model(batch)

# 打印结果
print(prof.key_averages().table(sort_by="npu_time_total", row_limit=20))

# 导出 Chrome Tracing 格式
prof.export_chrome_trace("trace.json")
```

### 输出示例

```
---------------------------------  --------  --------  --------  --------
Name                               CPU time  NPU time  # Calls   % NPU
---------------------------------  --------  --------  --------  --------
aten::conv2d                       0.123ms   2.456ms   50        35.2%
aten::batch_norm                   0.045ms   0.892ms   50        12.8%
aten::relu                         0.012ms   0.234ms   50         3.4%
aten::linear                       0.089ms   1.567ms   50        22.5%
aten::add                          0.008ms   0.123ms   100        1.8%
---------------------------------  --------  --------  --------  --------
```

### C++ API

```cpp
#include "acl/acl_prof.h"

// 初始化 Profiling
aclprofConfig* profConfig = aclprofCreateConfig(
    nullptr, 0,
    ACL_AICORE_PIPE_UTILIZATION,  // 采集 AI Core 流水线利用率
    nullptr,
    ACL_PROF_ACL_API | ACL_PROF_TASK_TIME | ACL_PROF_AICORE_METRICS
);

// 开始 Profiling
aclprofStart(profConfig);

// 执行推理
aclmdlExecute(modelId, inputDataset, outputDataset);
aclrtSynchronizeStream(stream);

// 停止 Profiling
aclprofStop(profConfig);

// 导出数据
aclprofFinalize();
aclprofDestroyConfig(profConfig);
```

---

## 性能瓶颈分析方法论

### 1. 确定瓶颈类型

```
计算密集型（Compute Bound）：
  AI Core 利用率 > 80%
  → 优化算子实现，使用融合算子

内存带宽密集型（Memory Bound）：
  HBM 带宽利用率 > 80%，AI Core 利用率低
  → 减少内存访问，使用算子融合，优化数据格式

通信密集型（Communication Bound）：
  HCCL 通信时间占比 > 30%
  → 重叠通信与计算，使用梯度压缩

CPU 瓶颈（CPU Bound）：
  NPU 等待 CPU 时间长
  → 使用数据下沉模式，减少 Host-Device 交互
```

### 2. 算子级分析

```python
# 找出耗时最长的算子
df = pd.read_csv("op_summary.csv")
slow_ops = df[df["Total Time(us)"] > 1000]  # 超过 1ms 的算子

for _, row in slow_ops.iterrows():
    print(f"Op: {row['Op Name']}")
    print(f"  Type: {row['Op Type']}")
    print(f"  Time: {row['Total Time(us)']}us")
    print(f"  Input shapes: {row['Input Shapes']}")
    print()
```

### 3. 时间线分析

在 Chrome 浏览器中打开 `chrome://tracing`，加载 `trace.json`：

```
时间线视图显示：
- 每个算子的开始/结束时间
- CPU 与 NPU 的并行情况
- 通信操作的时间
- 内存分配/释放事件
```

---

## 精度分析工具：msaccucmp

```bash
# 安装
pip install msaccucmp

# 开启 Dump（收集算子输入输出数据）
export ENABLE_DUMP=1
export DUMP_PATH=./npu_dump
export DUMP_STEP=0  # 第 0 步

# 运行推理
python inference.py

# 同样收集 CPU 参考数据
export ENABLE_DUMP=1
export DUMP_PATH=./cpu_dump
python inference_cpu.py

# 比对精度
python -m msaccucmp compare \
    -m ./npu_dump \
    -g ./cpu_dump \
    -o ./compare_result \
    --threshold 0.001  # 误差阈值
```

### 精度比对报告

```
比对结果：
算子名称              余弦相似度    最大绝对误差    状态
Conv2D_1             0.9999        0.0001         PASS
BatchNorm_1          0.9998        0.0003         PASS
Softmax_1            0.9985        0.0023         WARNING
LayerNorm_2          0.9921        0.0156         FAIL  ← 需要关注
```

---

## MindStudio 集成开发环境

MindStudio 是华为提供的 AI 开发 IDE，集成了：

- 代码编辑与调试
- 模型可视化
- Profiling 可视化分析
- 精度比对
- 算子开发工具

### 主要功能

```
模型分析：
  - 可视化计算图结构
  - 查看算子属性和连接关系
  - 识别潜在的性能问题

性能分析：
  - 时间线可视化
  - 算子耗时热力图
  - 内存使用趋势图
  - AI Core 利用率分析

精度分析：
  - 逐层精度比对
  - 异常值检测
  - 精度下降定位
```
