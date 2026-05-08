# Ascend C：新一代算子编程语言

## Ascend C 概述

Ascend C 是华为推出的新一代昇腾算子编程语言，基于标准 C++ 语法扩展，是 TBE TIK 的官方替代方案。

**核心优势**：
- 标准 C++ 语法，学习曲线低
- 支持 CPU 侧仿真调试，无需真实硬件
- 编译器自动优化，性能接近手写汇编
- 与 Triton/TileLang 等开源工具链兼容

---

## 编程模型

### 核函数（Kernel Function）

Ascend C 的核心是 `__global__` 修饰的核函数，运行在 AI Core 上：

```cpp
// 核函数声明
extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR x,      // 输入1（全局内存地址）
    GM_ADDR y,      // 输入2
    GM_ADDR z       // 输出
) {
    // 核函数实现
    KernelAdd op;
    op.Init(x, y, z);
    op.Process();
}
```

### 核函数调用（Host 侧）

```cpp
// Host 侧调用核函数
// 参数：<<<blockDim, l2ctrl, stream>>>
add_custom<<<blockDim, l2ctrl, stream>>>(x_gm, y_gm, z_gm);
```

---

## 核心 API 体系

### 内存层次 API

```cpp
// 全局内存（HBM）访问
GlobalTensor<half> xGlobal;
xGlobal.SetGlobalBuffer((__gm__ half*)x, totalLength);

// 本地内存（UB）声明
TBuf<TPosition::VECCALC> tmpBuf;  // 向量计算缓冲区
LocalTensor<half> xLocal = tmpBuf.Get<half>();
```

### 数据搬运 API

```cpp
// DataCopy：GM → UB（数据搬入）
DataCopy(xLocal, xGlobal[offset], copyLength);

// DataCopy：UB → GM（数据搬出）
DataCopy(zGlobal[offset], zLocal, copyLength);
```

### 向量计算 API

```cpp
// 逐元素加法
Add(zLocal, xLocal, yLocal, length);

// 逐元素乘法
Mul(zLocal, xLocal, yLocal, length);

// ReLU
Relu(zLocal, xLocal, length);

// 最大值
Max(zLocal, xLocal, yLocal, length);

// 数据类型转换
Cast(zLocalFP32, xLocalFP16, RoundMode::CAST_NONE, length);
```

### 矩阵计算 API

```cpp
// 矩阵乘法（Cube 单元）
Mmad(cLocal, aLocal, bLocal, M, K, N, false);
```

### 同步 API

```cpp
// 设置 Pipe 间同步标志
SetFlag<HardEvent::V_MTE2>(eventId);   // 向量计算完成通知 MTE2
WaitFlag<HardEvent::MTE2_V>(eventId);  // 等待 MTE2 完成

// 流水线同步
PipeBarrier<PIPE_ALL>();  // 等待所有 Pipe 完成
```

---

## 完整示例：Add 算子

```cpp
#include "kernel_operator.h"

// 算子实现类
class KernelAdd {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                 uint32_t totalLength, uint32_t tileLength) {
        // 保存参数
        this->totalLength = totalLength;
        this->tileLength = tileLength;
        this->tileNum = totalLength / tileLength;
        
        // 初始化全局内存张量
        xGm.SetGlobalBuffer((__gm__ half*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ half*)y, totalLength);
        zGm.SetGlobalBuffer((__gm__ half*)z, totalLength);
        
        // 初始化 Pipe（内存管理器）
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(half));
    }
    
    __aicore__ inline void Process() {
        // 分 Tile 处理
        for (uint32_t i = 0; i < tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress) {
        // 从 GM 搬入数据到 UB
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        
        DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        DataCopy(yLocal, yGm[progress * tileLength], tileLength);
        
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    
    __aicore__ inline void Compute(uint32_t progress) {
        // 执行向量加法
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        
        Add(zLocal, xLocal, yLocal, tileLength);
        
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    
    __aicore__ inline void CopyOut(uint32_t progress) {
        // 将结果从 UB 搬出到 GM
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * tileLength], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    // 全局内存张量
    GlobalTensor<half> xGm, yGm, zGm;
    
    // 队列（管理 UB 内存）
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    
    // Pipe（内存分配器）
    TPipe pipe;
    
    // 参数
    uint32_t totalLength, tileLength, tileNum;
    
    static constexpr int32_t BUFFER_NUM = 2;  // 双缓冲
};

// 核函数入口
extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling
) {
    GET_TILING_DATA(tilingData, tiling);
    KernelAdd op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileLength);
    op.Process();
}
```

---

## Tiling 机制

Tiling 是 Ascend C 算子开发的核心，决定如何将大数据分块处理。

### Tiling 参数定义

```cpp
// tiling.h - 定义 Tiling 数据结构
#include "register/tilingdata_base.h"

BEGIN_TILING_DATA_DEF(AddTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);  // 总元素数
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);   // 每个 Tile 的元素数
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);      // Tile 数量
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(AddCustom, AddTilingData)
```

### Tiling 计算函数（Host 侧）

```cpp
// tiling.cpp - 在 Host 侧计算 Tiling 参数
#include "tiling.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    AddTilingData tiling;
    
    // 获取输入形状
    auto x_shape = context->GetInputShape(0)->GetStorageShape();
    uint32_t totalLength = x_shape.GetShapeSize();
    
    // 计算 Tile 大小（根据 UB 容量）
    // UB 容量约 256KB，FP16 占 2 字节
    // 双缓冲需要 2 倍空间
    uint32_t ubSize = 256 * 1024;  // 256KB
    uint32_t tileLength = ubSize / sizeof(uint16_t) / 2 / 3;  // 3个队列
    tileLength = (tileLength / 128) * 128;  // 128 对齐
    
    tiling.set_totalLength(totalLength);
    tiling.set_tileLength(tileLength);
    tiling.set_tileNum(totalLength / tileLength);
    
    // 设置 block 数量（多核并行）
    context->SetBlockDim(8);  // 使用 8 个 AI Core
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
    return ge::GRAPH_SUCCESS;
}

REGISTER_OP_TILING_FUNC_BUFFERED(AddCustom, TilingFunc);

}  // namespace optiling
```

---

## 多核并行

Ascend C 支持多个 AI Core 并行执行：

```cpp
extern "C" __global__ __aicore__ void add_custom_multicore(
    GM_ADDR x, GM_ADDR y, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling
) {
    GET_TILING_DATA(tilingData, tiling);
    
    // 获取当前 Core 的 ID
    uint32_t coreId = GetBlockIdx();
    uint32_t coreNum = GetBlockNum();
    
    // 每个 Core 处理一部分数据
    uint32_t perCoreLength = tilingData.totalLength / coreNum;
    uint32_t offset = coreId * perCoreLength;
    
    KernelAdd op;
    op.Init(x + offset * sizeof(half),
            y + offset * sizeof(half),
            z + offset * sizeof(half),
            perCoreLength, tilingData.tileLength);
    op.Process();
}
```

---

## CPU 仿真调试

Ascend C 支持在 CPU 上仿真执行，方便调试：

```cpp
// cpu_main.cpp - CPU 仿真入口
#include "kernel_operator.h"

int main() {
    // 准备测试数据
    const int N = 1024;
    std::vector<half> x(N, 1.0f);
    std::vector<half> y(N, 2.0f);
    std::vector<half> z(N, 0.0f);
    
    // CPU 仿真调用
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    add_custom<<<1, nullptr, nullptr>>>(
        x.data(), y.data(), z.data(), nullptr, nullptr
    );
    
    // 验证结果
    for (int i = 0; i < N; i++) {
        assert(abs((float)z[i] - 3.0f) < 1e-3);
    }
    printf("CPU simulation passed!\n");
    return 0;
}
```

```bash
# 编译 CPU 仿真版本
cmake .. -DASCEND_COMPUTE_UNIT=cpu -DCMAKE_BUILD_TYPE=Debug
make -j4
./add_custom_test
```

---

## 项目结构

```
custom_add_op/
├── CMakeLists.txt
├── op_kernel/
│   └── add_custom.cpp      # Kernel 实现
├── op_host/
│   ├── add_custom_tiling.h # Tiling 数据结构
│   └── add_custom.cpp      # Tiling 计算 + 算子注册
└── test/
    ├── cpu_test.cpp         # CPU 仿真测试
    └── npu_test.cpp         # NPU 真实测试
```

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(add_custom)

set(ASCEND_PATH $ENV{ASCEND_TOOLKIT_HOME})

# Kernel 编译（NPU 侧）
add_custom_target(kernel
    COMMAND ${ASCEND_PATH}/bin/ccec_compiler
            --cce-aicore-arch=dav-c220
            -O2
            -o add_custom.o
            ${CMAKE_SOURCE_DIR}/op_kernel/add_custom.cpp
)

# Host 编译
add_library(add_custom_host SHARED op_host/add_custom.cpp)
target_include_directories(add_custom_host PRIVATE ${ASCEND_PATH}/include)
target_link_libraries(add_custom_host ${ASCEND_PATH}/lib64/libascendcl.so)
```
