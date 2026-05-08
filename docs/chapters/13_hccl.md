# HCCL 集合通信库

## HCCL 概述

HCCL（Huawei Collective Communication Library）是 CANN 提供的高性能集合通信库，专为昇腾 AI 处理器的分布式训练场景设计，是昇腾生态中对应 NVIDIA NCCL 的组件。

**核心特性**：
- 支持 12 种集合通信原语
- 拓扑感知的通信算法选择
- 支持单机多卡（HCCS 互联）和多机多卡（RoCE 网络）
- 与 PyTorch/MindSpore 深度集成

---

## 集合通信原语

### AllReduce

所有进程的数据进行规约（求和/最大值等），结果广播到所有进程。

```
进程0: [1,2,3]  ─┐
进程1: [4,5,6]  ─┤─→ AllReduce(SUM) ─→ 所有进程: [5,7,9]
进程2: [0,0,0]  ─┘
```

**用途**：梯度同步（数据并行训练的核心操作）

### AllGather

每个进程贡献一部分数据，所有进程收集完整数据。

```
进程0: [A]  ─┐
进程1: [B]  ─┤─→ AllGather ─→ 所有进程: [A,B,C]
进程2: [C]  ─┘
```

**用途**：收集分布式参数（张量并行）

### ReduceScatter

规约后将结果分散到各进程。

```
进程0: [1,2,3,4]  ─┐
进程1: [5,6,7,8]  ─┤─→ ReduceScatter(SUM) ─→ 进程0:[6], 进程1:[8], 进程2:[10], 进程3:[12]
进程2: [1,2,3,4]  ─┘
```

**用途**：ZeRO 优化器（减少内存占用）

### Broadcast

一个进程的数据广播到所有进程。

```
进程0: [data]  ─→ Broadcast(root=0) ─→ 所有进程: [data]
```

**用途**：参数初始化同步

### Scatter / Gather / Reduce

```
Scatter:  root 进程将数据分散到各进程
Gather:   各进程数据汇聚到 root 进程
Reduce:   各进程数据规约到 root 进程
```

### Send / Recv（点对点通信）

```c
// 发送数据
HcclSend(sendBuf, count, dataType, destRank, comm, stream);

// 接收数据
HcclRecv(recvBuf, count, dataType, srcRank, comm, stream);
```

---

## HCCL 初始化

### 方式一：使用 rank table 文件

```json
// hccl.json - rank table 配置文件
{
    "version": "1.0",
    "server_count": "2",
    "server_list": [
        {
            "server_id": "192.168.1.1",
            "device": [
                {"device_id": "0", "device_ip": "192.168.2.1", "rank_id": "0"},
                {"device_id": "1", "device_ip": "192.168.2.2", "rank_id": "1"},
                {"device_id": "2", "device_ip": "192.168.2.3", "rank_id": "2"},
                {"device_id": "3", "device_ip": "192.168.2.4", "rank_id": "3"}
            ]
        },
        {
            "server_id": "192.168.1.2",
            "device": [
                {"device_id": "0", "device_ip": "192.168.2.5", "rank_id": "4"},
                {"device_id": "1", "device_ip": "192.168.2.6", "rank_id": "5"},
                {"device_id": "2", "device_ip": "192.168.2.7", "rank_id": "6"},
                {"device_id": "3", "device_ip": "192.168.2.8", "rank_id": "7"}
            ]
        }
    ],
    "status": "completed"
}
```

```c
#include "hccl/hccl.h"

// 初始化 HCCL
HcclComm comm;
uint32_t rankId = 0;    // 当前进程的 rank
uint32_t rankSize = 8;  // 总进程数

HcclCommInitRootInfo(rankSize, &rootInfo, rankId, &comm);
```

### 方式二：使用 HcclCommInitAll（单机多卡）

```c
// 单机 8 卡初始化（最简单）
HcclComm comms[8];
int32_t deviceList[8] = {0, 1, 2, 3, 4, 5, 6, 7};
HcclCommInitAll(8, deviceList, comms);
```

---

## HCCL API 使用

### AllReduce 示例

```c
#include "hccl/hccl.h"
#include "acl/acl.h"

void allreduce_example() {
    // 初始化（假设已完成）
    HcclComm comm;
    aclrtStream stream;
    
    // 准备数据
    size_t count = 1024;
    void* sendBuf;
    void* recvBuf;
    aclrtMalloc(&sendBuf, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&recvBuf, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    
    // 执行 AllReduce（异步）
    HcclAllReduce(
        sendBuf,        // 发送缓冲区
        recvBuf,        // 接收缓冲区
        count,          // 元素数量
        HCCL_DATA_TYPE_FP32,  // 数据类型
        HCCL_REDUCE_SUM,      // 规约操作
        comm,           // 通信域
        stream          // 执行流
    );
    
    // 等待完成
    aclrtSynchronizeStream(stream);
    
    // 清理
    aclrtFree(sendBuf);
    aclrtFree(recvBuf);
}
```

### 支持的数据类型

```c
typedef enum {
    HCCL_DATA_TYPE_INT8   = 0,
    HCCL_DATA_TYPE_INT16  = 1,
    HCCL_DATA_TYPE_INT32  = 2,
    HCCL_DATA_TYPE_FP16   = 3,
    HCCL_DATA_TYPE_FP32   = 4,
    HCCL_DATA_TYPE_INT64  = 5,
    HCCL_DATA_TYPE_UINT64 = 6,
    HCCL_DATA_TYPE_UINT8  = 7,
    HCCL_DATA_TYPE_BF16   = 9,
} HcclDataType;
```

### 支持的规约操作

```c
typedef enum {
    HCCL_REDUCE_SUM  = 0,  // 求和
    HCCL_REDUCE_PROD = 1,  // 乘积
    HCCL_REDUCE_MAX  = 2,  // 最大值
    HCCL_REDUCE_MIN  = 3,  // 最小值
} HcclReduceOp;
```

---

## 通信拓扑与算法

HCCL 根据硬件拓扑自动选择最优通信算法：

### 单机 8 卡（HCCS 互联）

```
卡0 ─── 卡1 ─── 卡2 ─── 卡3
 │                         │
卡4 ─── 卡5 ─── 卡6 ─── 卡7

HCCS 全互联，带宽约 400GB/s
算法：Ring AllReduce 或 Tree AllReduce
```

### 多机多卡（RoCE 网络）

```
机器1: [卡0-7] ─── RoCE 100GbE ─── 机器2: [卡0-7]
                        │
                   机器3: [卡0-7]

节点内：HCCS（高带宽）
节点间：RoCE（相对低带宽）
算法：分层 AllReduce（先节点内，再节点间）
```

### Ring AllReduce 原理

```
N 个节点，数据大小 D：
1. ReduceScatter 阶段：每个节点发送 D/N 数据，共 N-1 轮
2. AllGather 阶段：每个节点广播 D/N 数据，共 N-1 轮
总通信量：2 × (N-1)/N × D ≈ 2D（与节点数无关！）
```

---

## 与 PyTorch 集成

```python
import torch
import torch.distributed as dist
import torch_npu

# 初始化分布式进程组（使用 HCCL 后端）
dist.init_process_group(
    backend="hccl",
    init_method="env://",
    world_size=8,
    rank=int(os.environ["RANK"])
)

# 设置当前设备
local_rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(local_rank)

# 模型并行化
model = MyModel().npu()
model = torch.nn.parallel.DistributedDataParallel(
    model, 
    device_ids=[local_rank]
)

# 训练循环（梯度同步自动通过 HCCL AllReduce 完成）
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()  # 自动触发 AllReduce
    optimizer.step()
```

### 启动分布式训练

```bash
# 单机 8 卡
torchrun --nproc_per_node=8 train.py

# 多机多卡（2机×8卡）
# 机器1
torchrun --nproc_per_node=8 \
         --nnodes=2 \
         --node_rank=0 \
         --master_addr=192.168.1.1 \
         --master_port=29500 \
         train.py

# 机器2
torchrun --nproc_per_node=8 \
         --nnodes=2 \
         --node_rank=1 \
         --master_addr=192.168.1.1 \
         --master_port=29500 \
         train.py
```

---

## HCCL 性能调优

### 1. 通信与计算重叠

```python
# 使用 no_sync() 累积梯度，减少通信频率
for i, batch in enumerate(dataloader):
    if i % accumulation_steps != 0:
        with model.no_sync():  # 不触发 AllReduce
            loss = model(batch)
            loss.backward()
    else:
        loss = model(batch)
        loss.backward()  # 触发 AllReduce
        optimizer.step()
        optimizer.zero_grad()
```

### 2. 梯度压缩

```python
# 使用 FP16 梯度通信（减少通信量 50%）
from torch.cuda.amp import GradScaler
scaler = GradScaler()

with torch.autocast(device_type='npu', dtype=torch.float16):
    loss = model(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. 环境变量调优

```bash
# 设置 HCCL 通信超时时间
export HCCL_TIMEOUT=1800  # 秒

# 设置通信缓冲区大小
export HCCL_BUFFSIZE=128  # MB

# 开启 HCCL 日志
export HCCL_LOG_LEVEL=INFO
```
