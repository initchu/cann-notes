# 驱动层与 Runtime 运行时

## NPU 驱动架构

昇腾 NPU 驱动是 CANN 软件栈的最底层，负责操作系统与硬件之间的交互。

```
┌─────────────────────────────────────────────────────┐
│                  用户态应用                          │
├─────────────────────────────────────────────────────┤
│              CANN Runtime（用户态）                  │
│         libascendcl.so / libruntime.so               │
├─────────────────────────────────────────────────────┤
│              驱动用户态库                            │
│              libdrvdsmi.so 等                        │
├─────────────────────────────────────────────────────┤
│              系统调用接口（ioctl）                    │
├─────────────────────────────────────────────────────┤
│              NPU 内核态驱动                          │
│              davinci_driver.ko                       │
├─────────────────────────────────────────────────────┤
│              PCIe / HCCS 总线                        │
├─────────────────────────────────────────────────────┤
│              昇腾 AI 处理器硬件                      │
└─────────────────────────────────────────────────────┘
```

### 驱动主要功能

- **设备枚举与初始化**：识别系统中的昇腾设备
- **内存管理**：HBM 内存的分配、释放、映射
- **任务调度**：将计算任务下发到硬件执行队列
- **中断处理**：处理硬件完成中断
- **错误恢复**：硬件异常检测与恢复

### 驱动安装验证

```bash
# 查看驱动版本
cat /proc/driver/npu/version

# 查看设备状态
npu-smi info

# 查看设备列表
npu-smi info -l

# 查看单个设备详情
npu-smi info -t common -i 0
```

---

## CANN Runtime 核心概念

Runtime 是 CANN 的运行时管理层，提供以下核心抽象：

### 1. Device（设备）

Device 代表一个物理昇腾 NPU。

```c
// 获取设备数量
uint32_t deviceCount;
aclrtGetDeviceCount(&deviceCount);

// 设置当前使用的设备
aclrtSetDevice(deviceId);  // deviceId 从 0 开始

// 重置设备（释放所有资源）
aclrtResetDevice(deviceId);
```

**关键点**：
- 每个进程可以使用多个设备
- 设备 ID 与物理 NPU 编号对应
- 使用完毕必须调用 `aclrtResetDevice` 释放资源

### 2. Context（上下文）

Context 是设备上的资源管理容器，类似 CUDA 的 `cudaContext`。

```c
aclrtContext context;

// 创建 Context（绑定到当前设备）
aclrtCreateContext(&context, deviceId);

// 设置当前线程的 Context
aclrtSetCurrentContext(context);

// 获取当前 Context
aclrtGetCurrentContext(&context);

// 销毁 Context
aclrtDestroyContext(context);
```

**Context 与线程的关系**：
- 每个线程有独立的当前 Context
- 多线程可以共享同一个 Context（需要同步）
- 推荐每个线程创建独立的 Context

### 3. Stream（流）

Stream 是任务执行的有序队列，类似 CUDA 的 `cudaStream_t`。

```c
aclrtStream stream;

// 创建 Stream
aclrtCreateStream(&stream);

// 在 Stream 上提交任务（异步）
aclrtMemcpyAsync(dst, dstSize, src, srcSize, kind, stream);
aclmdlExecuteAsync(modelId, input, output, stream);

// 等待 Stream 上所有任务完成
aclrtSynchronizeStream(stream);

// 销毁 Stream
aclrtDestroyStream(stream);
```

**Stream 的关键特性**：
- 同一 Stream 内的任务**顺序执行**
- 不同 Stream 之间的任务**并发执行**
- 通过 Event 实现跨 Stream 同步

### 4. Event（事件）

Event 用于精确的时间测量和跨 Stream 同步。

```c
aclrtEvent event;

// 创建 Event
aclrtCreateEvent(&event);

// 在 Stream 中记录 Event（异步）
aclrtRecordEvent(event, stream);

// 等待 Event 完成（CPU 侧阻塞）
aclrtSynchronizeEvent(event);

// 让 Stream 等待 Event（GPU 侧等待，CPU 不阻塞）
aclrtStreamWaitEvent(stream, event);

// 查询 Event 状态
aclrtEventStatus status;
aclrtQueryEvent(event, &status);

// 销毁 Event
aclrtDestroyEvent(event);
```

---

## 内存管理

### 内存类型

| 内存类型 | 分配函数 | 位置 | 特点 |
|----------|---------|------|------|
| Device 内存 | `aclrtMalloc` | HBM | NPU 直接访问，高带宽 |
| Host 内存（普通） | `malloc` / `new` | DDR | CPU 访问，不可 DMA |
| Host 内存（锁页） | `aclrtMallocHost` | DDR | 支持 DMA，传输更快 |

### 内存分配

```c
void* devPtr;
size_t size = 1024 * 1024;  // 1MB

// 分配 Device 内存
aclrtMalloc(&devPtr, size, ACL_MEM_MALLOC_HUGE_FIRST);

// 分配锁页 Host 内存
void* hostPtr;
aclrtMallocHost(&hostPtr, size);

// 释放
aclrtFree(devPtr);
aclrtFreeHost(hostPtr);
```

### 内存传输

```c
// 同步传输（阻塞直到完成）
aclrtMemcpy(dst, dstSize, src, srcSize, ACL_MEMCPY_HOST_TO_DEVICE);
aclrtMemcpy(dst, dstSize, src, srcSize, ACL_MEMCPY_DEVICE_TO_HOST);
aclrtMemcpy(dst, dstSize, src, srcSize, ACL_MEMCPY_DEVICE_TO_DEVICE);

// 异步传输（立即返回，在 Stream 上排队）
aclrtMemcpyAsync(dst, dstSize, src, srcSize, ACL_MEMCPY_HOST_TO_DEVICE, stream);
```

### 内存传输方向枚举

```c
typedef enum aclrtMemcpyKind {
    ACL_MEMCPY_HOST_TO_HOST,    // Host → Host
    ACL_MEMCPY_HOST_TO_DEVICE,  // Host → Device（上传）
    ACL_MEMCPY_DEVICE_TO_HOST,  // Device → Host（下载）
    ACL_MEMCPY_DEVICE_TO_DEVICE // Device → Device（设备内复制）
} aclrtMemcpyKind;
```

---

## 完整初始化/销毁流程

```c
#include "acl/acl.h"

int main() {
    // ===== 初始化阶段 =====
    
    // 1. 初始化 AscendCL（必须最先调用）
    aclInit(nullptr);  // nullptr 表示使用默认配置
    
    // 2. 指定设备
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    
    // 3. 创建 Context
    aclrtContext context;
    aclrtCreateContext(&context, deviceId);
    
    // 4. 创建 Stream
    aclrtStream stream;
    aclrtCreateStream(&stream);
    
    // ===== 业务逻辑 =====
    // ... 执行推理、内存操作等 ...
    
    // ===== 销毁阶段（逆序释放）=====
    
    // 4. 销毁 Stream
    aclrtDestroyStream(stream);
    
    // 3. 销毁 Context
    aclrtDestroyContext(context);
    
    // 2. 重置设备
    aclrtResetDevice(deviceId);
    
    // 1. 反初始化 AscendCL（必须最后调用）
    aclFinalize();
    
    return 0;
}
```

---

## 错误处理

CANN 所有 API 返回 `aclError` 类型的错误码：

```c
#include "acl/acl.h"

// 错误码检查宏
#define CHECK_ACL(call) \
    do { \
        aclError ret = (call); \
        if (ret != ACL_SUCCESS) { \
            printf("ACL error: %d at %s:%d\n", ret, __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

// 使用示例
CHECK_ACL(aclInit(nullptr));
CHECK_ACL(aclrtSetDevice(0));

// 获取错误描述
const char* errMsg = aclGetRecentErrMsg();
printf("Error: %s\n", errMsg);
```

### 常见错误码

| 错误码 | 含义 | 常见原因 |
|--------|------|----------|
| `ACL_SUCCESS` (0) | 成功 | - |
| `ACL_ERROR_INVALID_PARAM` | 参数无效 | 空指针、越界 |
| `ACL_ERROR_MEMORY_ADDRESS_UNALIGNED` | 内存未对齐 | 地址不满足对齐要求 |
| `ACL_ERROR_DEVICE_NOT_EXIST` | 设备不存在 | deviceId 超出范围 |
| `ACL_ERROR_REPEAT_INITIALIZE` | 重复初始化 | aclInit 调用多次 |

---

## 多线程使用模式

### 模式一：每线程独立 Context

```c
// 推荐模式：每个工作线程有独立的 Context 和 Stream
void worker_thread(int deviceId) {
    aclrtContext ctx;
    aclrtCreateContext(&ctx, deviceId);
    
    aclrtStream stream;
    aclrtCreateStream(&stream);
    
    // 执行推理任务...
    
    aclrtDestroyStream(stream);
    aclrtDestroyContext(ctx);
}
```

### 模式二：共享 Context，独立 Stream

```c
// 多线程共享 Context，但每个线程有独立 Stream
// 需要在每个线程中调用 aclrtSetCurrentContext
void worker_thread(aclrtContext sharedCtx) {
    aclrtSetCurrentContext(sharedCtx);
    
    aclrtStream stream;
    aclrtCreateStream(&stream);
    
    // 执行任务...
    
    aclrtDestroyStream(stream);
}
```

---

## npu-smi 监控工具

```bash
# 实时监控（类似 watch nvidia-smi）
watch -n 1 npu-smi info

# 查看内存使用
npu-smi info -t usages -i 0

# 查看温度和功耗
npu-smi info -t board -i 0

# 查看进程占用
npu-smi info -t proc -i 0
```

输出示例：
```
+-----------------------------------------------------------------------------------+
| npu-smi 23.0.0                                 Time : 2026-04-16 10:00:00         |
+-----------------------------------------------------------------------------------+
| NPU   Name         Health   Power(W)   Temp(C)   Hugepages-Usage(page)            |
| Chip                        Bus-Id     AICore(%)  Memory-Usage(MB)                |
+===================================================================================+
| 0     910B3        OK       310.0      45         0    / 0                        |
| 0                           0000:C1:00.0  0       1024 / 65536                   |
+-----------------------------------------------------------------------------------+
```
