# AscendCL 内存管理深度解析

## 内存模型概述

昇腾平台的内存管理是 CANN 开发中最需要深入理解的部分之一。与 CPU 编程不同，昇腾平台存在明确的 Host（主机）和 Device（设备）内存空间分离。

```
┌─────────────────────────────────────────────────────────┐
│                    Host 侧（CPU）                        │
│                                                         │
│  普通内存（malloc）    锁页内存（aclrtMallocHost）         │
│  CPU 可访问           CPU 可访问 + 支持 DMA              │
│  不可 DMA             传输效率更高                        │
└──────────────────────────┬──────────────────────────────┘
                           │ PCIe / HCCS 总线
                           │ aclrtMemcpy / aclrtMemcpyAsync
┌──────────────────────────▼──────────────────────────────┐
│                    Device 侧（NPU）                      │
│                                                         │
│  HBM 内存（aclrtMalloc）                                │
│  NPU 直接访问，高带宽（~900GB/s）                        │
│  CPU 不可直接访问                                        │
└─────────────────────────────────────────────────────────┘
```

---

## 内存分配策略

### aclrtMalloc 分配策略

```c
typedef enum aclrtMemMallocPolicy {
    ACL_MEM_MALLOC_HUGE_FIRST,      // 优先分配大页内存（推荐）
    ACL_MEM_MALLOC_HUGE_ONLY,       // 仅分配大页内存
    ACL_MEM_MALLOC_NORMAL_ONLY,     // 仅分配普通内存
    ACL_MEM_MALLOC_HUGE_FIRST_P2P,  // 大页优先 + P2P 访问
    ACL_MEM_MALLOC_NORMAL_ONLY_P2P, // 普通内存 + P2P 访问
} aclrtMemMallocPolicy;
```

**推荐使用 `ACL_MEM_MALLOC_HUGE_FIRST`**：
- 大页内存（Huge Page）减少 TLB Miss
- 提升内存访问效率
- 分配失败时自动回退到普通内存

### 内存对齐

```c
// 昇腾对内存对齐有要求，建议按 512 字节对齐
size_t alignedSize = (size + 511) & ~511;
void* devPtr;
aclrtMalloc(&devPtr, alignedSize, ACL_MEM_MALLOC_HUGE_FIRST);
```

---

## 内存传输详解

### 同步 vs 异步传输

```c
// 同步传输：阻塞 CPU 直到传输完成
aclrtMemcpy(dst, dstSize, src, srcSize, ACL_MEMCPY_HOST_TO_DEVICE);

// 异步传输：立即返回，传输在 Stream 上排队
// 注意：src 必须是锁页内存（aclrtMallocHost 分配）
aclrtMemcpyAsync(dst, dstSize, src, srcSize, 
                  ACL_MEMCPY_HOST_TO_DEVICE, stream);
// 等待传输完成
aclrtSynchronizeStream(stream);
```

### 性能对比

| 传输方式 | Host 内存类型 | 特点 |
|----------|-------------|------|
| 同步 + 普通内存 | malloc | 最慢，CPU 阻塞 |
| 同步 + 锁页内存 | aclrtMallocHost | 较快，CPU 阻塞 |
| 异步 + 锁页内存 | aclrtMallocHost | 最快，可与计算重叠 |

### 零拷贝（Zero-Copy）

对于频繁小数据传输，可以使用零拷贝技术：

```c
// 分配可被 NPU 直接访问的 Host 内存（零拷贝）
void* hostPtr;
aclrtMallocHost(&hostPtr, size);

// 获取对应的 Device 虚拟地址
void* devVirtualPtr;
aclrtGetMemInfo(ACL_HBM_MEM, &freeSize, &totalSize);
// 注意：零拷贝性能取决于 PCIe 带宽，不适合大数据量
```

---

## 内存池管理

频繁的 `aclrtMalloc/aclrtFree` 会产生性能开销，生产环境推荐使用内存池：

```cpp
class DeviceMemoryPool {
public:
    DeviceMemoryPool(size_t blockSize, size_t blockCount) 
        : blockSize_(blockSize) {
        for (size_t i = 0; i < blockCount; i++) {
            void* ptr;
            aclrtMalloc(&ptr, blockSize, ACL_MEM_MALLOC_HUGE_FIRST);
            freeBlocks_.push(ptr);
        }
    }
    
    void* Allocate() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (freeBlocks_.empty()) return nullptr;
        void* ptr = freeBlocks_.front();
        freeBlocks_.pop();
        return ptr;
    }
    
    void Free(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        freeBlocks_.push(ptr);
    }
    
    ~DeviceMemoryPool() {
        while (!freeBlocks_.empty()) {
            aclrtFree(freeBlocks_.front());
            freeBlocks_.pop();
        }
    }

private:
    size_t blockSize_;
    std::queue<void*> freeBlocks_;
    std::mutex mutex_;
};
```

---

## 内存复用策略

### 输入输出内存复用

```c
// 如果输入数据在推理后不再需要，可以复用输入内存作为中间缓冲
// 但需要确保推理完成后再复用

// 推理完成
aclrtSynchronizeStream(stream);

// 复用输入内存存储其他数据
aclrtMemcpy(inputDevPtr, newDataSize, newData, newDataSize,
             ACL_MEMCPY_HOST_TO_DEVICE);
```

### 多模型内存共享

```c
// 多个模型共享同一块输出内存（串行推理时）
void* sharedOutputPtr;
size_t maxOutputSize = max(model1OutputSize, model2OutputSize);
aclrtMalloc(&sharedOutputPtr, maxOutputSize, ACL_MEM_MALLOC_HUGE_FIRST);

// 模型1推理
aclDataBuffer* buf1 = aclCreateDataBuffer(sharedOutputPtr, model1OutputSize);
// ... 推理 ...
aclrtSynchronizeStream(stream);
// 处理模型1输出

// 模型2推理（复用同一块内存）
aclDataBuffer* buf2 = aclCreateDataBuffer(sharedOutputPtr, model2OutputSize);
// ... 推理 ...
```

---

## 内存查询与监控

```c
// 查询 Device 内存使用情况
size_t freeSize, totalSize;
aclrtGetMemInfo(ACL_HBM_MEM, &freeSize, &totalSize);
printf("HBM: free=%zuMB, total=%zuMB\n", 
       freeSize/1024/1024, totalSize/1024/1024);

// 查询 Host 内存信息
aclrtGetMemInfo(ACL_DDR_MEM, &freeSize, &totalSize);
```

---

## 常见内存问题与排查

### 1. 内存泄漏

```c
// 错误：忘记释放 Device 内存
void* ptr;
aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
// ... 使用 ptr ...
// 忘记调用 aclrtFree(ptr) ← 内存泄漏！

// 正确：使用 RAII 封装
class DeviceBuffer {
    void* ptr_ = nullptr;
    size_t size_ = 0;
public:
    DeviceBuffer(size_t size) : size_(size) {
        aclrtMalloc(&ptr_, size, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ~DeviceBuffer() { if (ptr_) aclrtFree(ptr_); }
    void* get() { return ptr_; }
    size_t size() { return size_; }
};
```

### 2. 内存越界

```c
// 错误：写入超出分配大小
void* ptr;
aclrtMalloc(&ptr, 1024, ACL_MEM_MALLOC_HUGE_FIRST);
// 实际数据 2048 字节，越界写入！
aclrtMemcpy(ptr, 2048, src, 2048, ACL_MEMCPY_HOST_TO_DEVICE);

// 正确：确保目标大小足够
size_t dataSize = 2048;
aclrtMalloc(&ptr, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(ptr, dataSize, src, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
```

### 3. 使用已释放内存

```c
// 错误：释放后继续使用
aclrtFree(ptr);
aclrtMemcpy(dst, size, ptr, size, ACL_MEMCPY_DEVICE_TO_DEVICE); // 危险！

// 正确：释放后置空
aclrtFree(ptr);
ptr = nullptr;
```

### 4. Host/Device 内存混用

```c
// 错误：将普通 Host 内存地址传给 Device 操作
int* hostArray = new int[1024];
// 以下操作会失败，因为 hostArray 不是 Device 内存
aclrtMemcpy(devicePtr, size, hostArray, size, 
             ACL_MEMCPY_DEVICE_TO_DEVICE); // 错误！

// 正确：先拷贝到 Device
aclrtMemcpy(devicePtr, size, hostArray, size,
             ACL_MEMCPY_HOST_TO_DEVICE); // 正确
```

---

## 内存带宽优化技巧

### 1. 批量传输

```c
// 低效：多次小数据传输
for (int i = 0; i < 1000; i++) {
    aclrtMemcpy(dst + i*4, 4, src + i*4, 4, ACL_MEMCPY_HOST_TO_DEVICE);
}

// 高效：一次大数据传输
aclrtMemcpy(dst, 4000, src, 4000, ACL_MEMCPY_HOST_TO_DEVICE);
```

### 2. 传输与计算重叠

```c
aclrtStream computeStream, transferStream;
aclrtCreateStream(&computeStream);
aclrtCreateStream(&transferStream);

aclrtEvent transferDone;
aclrtCreateEvent(&transferDone);

// 在 transferStream 上传输下一批数据
aclrtMemcpyAsync(nextInputDev, size, nextInputHost, size,
                  ACL_MEMCPY_HOST_TO_DEVICE, transferStream);
aclrtRecordEvent(transferDone, transferStream);

// 在 computeStream 上执行当前批次推理
aclmdlExecuteAsync(modelId, currentInput, currentOutput, computeStream);

// computeStream 等待传输完成后才能使用新数据
aclrtStreamWaitEvent(computeStream, transferDone);

// 同步
aclrtSynchronizeStream(computeStream);
```
