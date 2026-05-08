# TBE 张量加速引擎

## TBE 概述

TBE（Tensor Boost Engine）是 CANN 的算子开发框架，提供两种开发模式：

| 模式 | 语言 | 特点 | 适用场景 |
|------|------|------|----------|
| **DSL 模式** | Python | 声明式，自动调度 | 规则算子，快速开发 |
| **TIK 模式** | Python | 命令式，手动控制 | 复杂算子，性能优先 |

```{note}
TBE TIK 模式已被 Ascend C 逐步替代。新项目推荐使用 Ascend C。
TBE DSL 模式仍然有效，适合快速开发规则算子。
```

---

## TBE DSL 模式

DSL（Domain Specific Language）模式使用类似 NumPy 的声明式语法描述算子计算逻辑，TBE 自动完成调度优化。

### 核心 API

```python
from te import tvm
from te.platform.fusion_manager import fusion_manager
import te.lang.cce as tbe
from topi import generic
```

### 示例：实现 ReLU 算子

```python
from te import tvm
import te.lang.cce as tbe
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

@fusion_manager.register("relu")
def relu_compute(input_x, output_y, kernel_name="relu"):
    """
    ReLU 算子计算函数
    y = max(x, 0)
    """
    # 使用 TBE DSL 描述计算
    res = tbe.vrelu(input_x)
    return res

def relu(input_x, output_y, kernel_name="relu"):
    """
    ReLU 算子入口函数
    """
    # 获取输入形状和数据类型
    shape = input_x.get("shape")
    dtype = input_x.get("dtype").lower()
    
    # 输入校验
    util.check_shape_rule(shape)
    util.check_dtype_rule(dtype, ["float16", "float32"])
    
    # 创建 TVM 占位符
    data = tvm.placeholder(shape, name="data", dtype=dtype)
    
    # 调用计算函数
    res = relu_compute(data, output_y, kernel_name)
    
    # 自动调度
    with tvm.target.cce():
        schedule = generic.auto_schedule(res)
    
    # 编译配置
    config = {"name": kernel_name, "tensor_list": [data, res]}
    
    # 编译生成二进制
    tbe.cce_build_code(schedule, config)
```

### 示例：实现 Add 算子

```python
from te import tvm
import te.lang.cce as tbe
from te.platform.fusion_manager import fusion_manager
from topi import generic

@fusion_manager.register("add")
def add_compute(input_x, input_y, output_z, kernel_name="add"):
    """
    逐元素加法：z = x + y
    """
    res = tbe.vadd(input_x, input_y)
    return res

def add(input_x, input_y, output_z, kernel_name="add"):
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    dtype = input_x.get("dtype").lower()
    
    # 广播处理
    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
    
    data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype)
    data_y = tvm.placeholder(shape_y, name="data_y", dtype=dtype)
    
    # 广播到相同形状
    data_x_broadcast = tbe.broadcast(data_x, shape_max)
    data_y_broadcast = tbe.broadcast(data_y, shape_max)
    
    res = add_compute(data_x_broadcast, data_y_broadcast, output_z, kernel_name)
    
    with tvm.target.cce():
        schedule = generic.auto_schedule(res)
    
    config = {"name": kernel_name, "tensor_list": [data_x, data_y, res]}
    tbe.cce_build_code(schedule, config)
```

### TBE DSL 常用 API

```python
# 向量运算
tbe.vadd(x, y)      # 逐元素加
tbe.vsub(x, y)      # 逐元素减
tbe.vmul(x, y)      # 逐元素乘
tbe.vdiv(x, y)      # 逐元素除
tbe.vrelu(x)        # ReLU
tbe.vexp(x)         # 指数
tbe.vlog(x)         # 对数
tbe.vsqrt(x)        # 平方根
tbe.vabs(x)         # 绝对值

# 归约运算
tbe.sum(x, axis)    # 求和
tbe.reduce_max(x, axis)  # 最大值
tbe.reduce_min(x, axis)  # 最小值

# 矩阵运算
tbe.matmul(x, y, trans_a=False, trans_b=False)  # 矩阵乘

# 广播
tbe.broadcast(x, shape)  # 广播到目标形状

# 数据类型转换
tbe.cast_to(x, dtype)    # 类型转换
```

---

## TBE TIK 模式

TIK（Tensor Iterator Kernel）提供更底层的控制，开发者需要显式管理数据搬运和计算流程。

### TIK 核心概念

```python
from te import tik

# 创建 TIK 实例
tik_instance = tik.Tik()

# 声明张量（在不同内存层次）
# GM：全局内存（HBM）
input_gm = tik_instance.Tensor("float16", (1024,), name="input", scope=tik.scope_gm)
output_gm = tik_instance.Tensor("float16", (1024,), name="output", scope=tik.scope_gm)

# UB：统一缓冲区（AI Core 本地）
input_ub = tik_instance.Tensor("float16", (256,), name="input_ub", scope=tik.scope_ubuf)
output_ub = tik_instance.Tensor("float16", (256,), name="output_ub", scope=tik.scope_ubuf)
```

### TIK 数据搬运

```python
# GM → UB（数据搬入）
tik_instance.data_move(
    input_ub,    # 目标（UB）
    input_gm,    # 源（GM）
    0,           # sid（通常为0）
    1,           # nburst（搬运次数）
    256 // 16,   # burst_len（每次搬运的 block 数，1 block = 32 字节）
    0,           # src_stride（源步长）
    0            # dst_stride（目标步长）
)

# UB → GM（数据搬出）
tik_instance.data_move(
    output_gm,
    output_ub,
    0, 1, 256 // 16, 0, 0
)
```

### TIK 向量计算

```python
# 向量加法
tik_instance.vec_add(
    128,         # mask（每次处理的元素数）
    output_ub,   # 目标
    input_ub,    # 源1
    input_ub,    # 源2
    1,           # repeat_times（重复次数）
    8, 8, 8      # dst_stride, src0_stride, src1_stride
)

# 向量乘法
tik_instance.vec_mul(128, output_ub, input_ub, weight_ub, 1, 8, 8, 8)

# 向量 ReLU
tik_instance.vec_relu(128, output_ub, input_ub, 1, 8, 8)
```

### TIK 矩阵计算（Cube）

```python
# 矩阵乘法（需要数据在 L0A/L0B 中）
# 先将数据从 UB 搬到 L0
input_l0a = tik_instance.Tensor("float16", (16, 16), scope=tik.scope_l0a)
weight_l0b = tik_instance.Tensor("float16", (16, 16), scope=tik.scope_l0b)
output_l0c = tik_instance.Tensor("float32", (16, 16), scope=tik.scope_l0c)

# 执行矩阵乘
tik_instance.mmad(output_l0c, input_l0a, weight_l0b, 16, 16, 16, False)
```

---

## 算子信息定义文件

每个算子需要一个 `op_info_cfg.py` 文件描述其元信息：

```python
# custom_relu_op_info_cfg.py
from op_test_frame.ut import OpInfo

op_info = OpInfo(
    op_type="CustomRelu",
    inputs=[
        {"name": "x", "dtype": ["float16", "float32"], "format": ["ND", "NCHW"]}
    ],
    outputs=[
        {"name": "y", "dtype": ["float16", "float32"], "format": ["ND", "NCHW"]}
    ],
    attr=[],
    kernel_name="custom_relu"
)
```

---

## 算子编译与测试

### 编译算子

```bash
# 使用 ATC 编译单算子
atc --singleop=op_list.json \
    --soc_version=Ascend910B3 \
    --output=./output

# op_list.json 格式
[{
    "op": "CustomRelu",
    "input_desc": [{"format": ["ND"], "type": "float16", "shape": [1, 1024]}],
    "output_desc": [{"format": ["ND"], "type": "float16", "shape": [1, 1024]}]
}]
```

### 单元测试

```python
# 使用 op_test_frame 进行单元测试
from op_test_frame.ut import OpUT

ut = OpUT("CustomRelu", "custom_relu", "custom_relu")

# 添加测试用例
ut.add_case(
    params=[
        {"shape": (1, 1024), "dtype": "float16", "format": "ND", "ori_shape": (1, 1024)},
        {"shape": (1, 1024), "dtype": "float16", "format": "ND", "ori_shape": (1, 1024)}
    ],
    case_name="test_relu_fp16"
)

# 运行测试
ut.run("test_relu_fp16")
```
