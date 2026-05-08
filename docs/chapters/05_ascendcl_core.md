# AscendCL 核心开发接口

## AscendCL 概述

AscendCL（Ascend Computing Language）是 CANN 提供给应用开发者的核心 C/C++ API 库。它封装了底层硬件细节，提供统一的编程接口，是开发昇腾 AI 应用的主要入口。

**AscendCL 的核心能力**：
- 设备/上下文/流管理
- 内存管理与数据传输
- 模型加载与推理执行
- 单算子调用
- 媒体数据处理（DVPP）

---

## AscendCL 数据类型体系

### 基础数据类型

```c
// 数据类型枚举
typedef enum aclDataType {
    ACL_DT_UNDEFINED = -1,
    ACL_FLOAT16 = 1,    // FP16
    ACL_FLOAT = 0,      // FP32
    ACL_INT8 = 2,       // INT8
    ACL_INT16 = 6,      // INT16
    ACL_INT32 = 3,      // INT32
    ACL_INT64 = 9,      // INT64
    ACL_UINT8 = 4,      // UINT8
    ACL_UINT16 = 7,     // UINT16
    ACL_UINT32 = 8,     // UINT32
    ACL_UINT64 = 10,    // UINT64
    ACL_BOOL = 12,      // BOOL
    ACL_BF16 = 27,      // BF16
} aclDataType;
```

### 内存格式（Format）

```c
typedef enum aclFormat {
    ACL_FORMAT_UNDEFINED = -1,
    ACL_FORMAT_NCHW = 0,    // 标准格式：Batch×Channel×Height×Width
    ACL_FORMAT_NHWC = 1,    // TF 常用格式
    ACL_FORMAT_ND = 2,      // 通用 N 维格式
    ACL_FORMAT_NC1HWC0 = 3, // 昇腾内部优化格式（5D）
    ACL_FORMAT_FRACTAL_Z = 4,// 昇腾矩阵优化格式
    ACL_FORMAT_NC1HWC0_C04 = 12,
    ACL_FORMAT_HWCN = 16,
    ACL_FORMAT_NDHWC = 27,
    ACL_FORMAT_FRACTAL_NZ = 29, // 昇腾 NZ 格式（矩阵乘法优化）
} aclFormat;
```

**关键格式说明**：
- `ACL_FORMAT_NCHW`：PyTorch 默认格式，推理时常用
- `ACL_FORMAT_NC1HWC0`：昇腾内部 5D 格式，C 维度按 16 对齐分块
- `ACL_FORMAT_FRACTAL_NZ`：矩阵乘法优化格式，Cube Unit 直接消费

---

## Tensor 数据结构

### aclTensorDesc：张量描述符

描述张量的形状、数据类型、格式等元信息（不包含数据本身）。

```c
// 创建张量描述符
// 参数：数据类型，维度数，各维度大小，内存格式
aclTensorDesc* inputDesc = aclCreateTensorDesc(
    ACL_FLOAT16,    // 数据类型
    4,              // 维度数
    dims,           // 各维度大小数组 [1, 3, 224, 224]
    ACL_FORMAT_NCHW // 内存格式
);

// 设置动态维度（-1 表示动态）
int64_t dims[] = {-1, 3, 224, 224};  // batch 维度动态

// 销毁描述符
aclDestroyTensorDesc(inputDesc);
```

### aclDataBuffer：数据缓冲区

持有实际的内存指针和大小。

```c
// 创建数据缓冲区（绑定已分配的 Device 内存）
void* devPtr;
aclrtMalloc(&devPtr, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);

aclDataBuffer* dataBuffer = aclCreateDataBuffer(devPtr, dataSize);

// 获取缓冲区信息
void* ptr = aclGetDataBufferAddr(dataBuffer);
size_t size = aclGetDataBufferSizeV2(dataBuffer);

// 销毁（不会释放 devPtr，需要单独 aclrtFree）
aclDestroyDataBuffer(dataBuffer);
```

---

## 模型推理完整示例

以下是一个完整的 AscendCL 推理应用示例：

```c
#include "acl/acl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 错误检查宏
#define CHECK_ACL(call) \
    do { \
        aclError ret = (call); \
        if (ret != ACL_SUCCESS) { \
            printf("[ERROR] %s failed, error code: %d\n", #call, ret); \
            return -1; \
        } \
    } while(0)

int main() {
    // ===== 1. 初始化 =====
    CHECK_ACL(aclInit(nullptr));
    
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    
    aclrtContext context;
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));
    
    // ===== 2. 加载模型 =====
    uint32_t modelId;
    CHECK_ACL(aclmdlLoadFromFile("model.om", &modelId));
    
    // 获取模型描述（输入输出信息）
    aclmdlDesc* modelDesc = aclmdlCreateDesc();
    CHECK_ACL(aclmdlGetDesc(modelDesc, modelId));
    
    // 查询输入输出数量
    size_t inputCount = aclmdlGetNumInputs(modelDesc);
    size_t outputCount = aclmdlGetNumOutputs(modelDesc);
    printf("Model inputs: %zu, outputs: %zu\n", inputCount, outputCount);
    
    // ===== 3. 准备输入数据集 =====
    aclmdlDataset* inputDataset = aclmdlCreateDataset();
    
    for (size_t i = 0; i < inputCount; i++) {
        size_t inputSize = aclmdlGetInputSizeByIndex(modelDesc, i);
        
        // 分配 Device 内存
        void* inputDevPtr;
        aclrtMalloc(&inputDevPtr, inputSize, ACL_MEM_MALLOC_HUGE_FIRST);
        
        // 准备输入数据（从 Host 拷贝）
        void* inputHostPtr;
        aclrtMallocHost(&inputHostPtr, inputSize);
        // ... 填充 inputHostPtr 数据 ...
        aclrtMemcpy(inputDevPtr, inputSize, inputHostPtr, inputSize,
                    ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtFreeHost(inputHostPtr);
        
        // 创建数据缓冲区并添加到数据集
        aclDataBuffer* inputBuffer = aclCreateDataBuffer(inputDevPtr, inputSize);
        aclmdlAddDatasetBuffer(inputDataset, inputBuffer);
    }
    
    // ===== 4. 准备输出数据集 =====
    aclmdlDataset* outputDataset = aclmdlCreateDataset();
    
    for (size_t i = 0; i < outputCount; i++) {
        size_t outputSize = aclmdlGetOutputSizeByIndex(modelDesc, i);
        
        void* outputDevPtr;
        aclrtMalloc(&outputDevPtr, outputSize, ACL_MEM_MALLOC_HUGE_FIRST);
        
        aclDataBuffer* outputBuffer = aclCreateDataBuffer(outputDevPtr, outputSize);
        aclmdlAddDatasetBuffer(outputDataset, outputBuffer);
    }
    
    // ===== 5. 执行推理 =====
    CHECK_ACL(aclmdlExecute(modelId, inputDataset, outputDataset));
    
    // ===== 6. 获取输出结果 =====
    for (size_t i = 0; i < outputCount; i++) {
        aclDataBuffer* outputBuffer = aclmdlGetDatasetBuffer(outputDataset, i);
        void* outputDevPtr = aclGetDataBufferAddr(outputBuffer);
        size_t outputSize = aclGetDataBufferSizeV2(outputBuffer);
        
        // 拷贝回 Host
        void* outputHostPtr;
        aclrtMallocHost(&outputHostPtr, outputSize);
        aclrtMemcpy(outputHostPtr, outputSize, outputDevPtr, outputSize,
                    ACL_MEMCPY_DEVICE_TO_HOST);
        
        // 处理输出数据...
        float* result = (float*)outputHostPtr;
        printf("Output[0] first value: %f\n", result[0]);
        
        aclrtFreeHost(outputHostPtr);
    }
    
    // ===== 7. 资源释放 =====
    // 释放数据集
    for (size_t i = 0; i < inputCount; i++) {
        aclDataBuffer* buf = aclmdlGetDatasetBuffer(inputDataset, i);
        void* ptr = aclGetDataBufferAddr(buf);
        aclrtFree(ptr);
        aclDestroyDataBuffer(buf);
    }
    aclmdlDestroyDataset(inputDataset);
    
    for (size_t i = 0; i < outputCount; i++) {
        aclDataBuffer* buf = aclmdlGetDatasetBuffer(outputDataset, i);
        void* ptr = aclGetDataBufferAddr(buf);
        aclrtFree(ptr);
        aclDestroyDataBuffer(buf);
    }
    aclmdlDestroyDataset(outputDataset);
    
    // 卸载模型
    aclmdlDestroyDesc(modelDesc);
    aclmdlUnload(modelId);
    
    // 销毁 Runtime 资源
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    
    return 0;
}
```

---

## 动态输入推理

当模型支持动态 batch 或动态分辨率时，需要在推理前设置实际输入形状：

```c
// 动态 Batch 推理
aclmdlSetDynamicBatchSize(modelId, inputDataset, 
                           index,    // 动态 batch 输入的索引
                           batchSize // 实际 batch 大小
                           );

// 动态分辨率推理
aclmdlSetDynamicHWSize(modelId, inputDataset,
                        index,   // 动态分辨率输入的索引
                        height,  // 实际高度
                        width    // 实际宽度
                        );

// 动态维度推理（最灵活）
aclmdlIODims dims;
dims.dimCount = 4;
dims.dims[0] = 1;    // batch
dims.dims[1] = 3;    // channel
dims.dims[2] = 224;  // height
dims.dims[3] = 224;  // width
aclmdlSetInputDynamicDims(modelId, inputDataset, index, &dims);
```

---

## 异步推理模式

异步推理可以实现推理与数据预处理的流水线并行：

```c
// 异步执行推理（立即返回）
aclmdlExecuteAsync(modelId, inputDataset, outputDataset, stream);

// 继续执行其他操作（如准备下一批数据）
prepare_next_batch();

// 等待推理完成
aclrtSynchronizeStream(stream);

// 处理输出结果
process_output(outputDataset);
```

---

## 单算子调用

AscendCL 支持直接调用单个算子（无需完整模型）：

```c
// 创建算子执行器
aclopAttr* opAttr = aclopCreateAttr();
aclopSetAttrBool(opAttr, "transpose_x1", false);
aclopSetAttrBool(opAttr, "transpose_x2", false);

// 准备输入输出描述符
aclTensorDesc* inputDescs[2] = {desc1, desc2};
aclDataBuffer* inputBuffers[2] = {buf1, buf2};
aclTensorDesc* outputDescs[1] = {outputDesc};
aclDataBuffer* outputBuffers[1] = {outputBuf};

// 执行算子
aclopExecuteV2(
    "MatMul",       // 算子类型名
    2,              // 输入数量
    inputDescs,     // 输入描述符数组
    inputBuffers,   // 输入数据缓冲区数组
    1,              // 输出数量
    outputDescs,    // 输出描述符数组
    outputBuffers,  // 输出数据缓冲区数组
    opAttr,         // 算子属性
    stream          // 执行流
);

aclrtSynchronizeStream(stream);
aclopDestroyAttr(opAttr);
```

---

## 编译与链接

```cmake
# CMakeLists.txt 示例
cmake_minimum_required(VERSION 3.14)
project(ascendcl_demo)

set(CMAKE_CXX_STANDARD 14)

# 设置 CANN 路径
set(ASCEND_PATH /usr/local/Ascend/ascend-toolkit/latest)

include_directories(${ASCEND_PATH}/include)
link_directories(${ASCEND_PATH}/lib64)

add_executable(demo main.cpp)
target_link_libraries(demo ascendcl acl_op_compiler)
```

```bash
# 编译
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# 运行（需要设置环境变量）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./demo
```
