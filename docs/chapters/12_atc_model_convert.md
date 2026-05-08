# ATC 模型转换工具

## ATC 概述

ATC（Ascend Tensor Compiler）是 CANN 提供的离线模型转换工具，将主流 AI 框架的模型文件转换为昇腾硬件可执行的 `.om`（Offline Model）格式。

**转换过程中 ATC 完成**：
- 图优化（算子融合、常量折叠等）
- 算子编译（生成 AI Core 二进制）
- 内存规划
- 生成可直接在昇腾上执行的模型

---

## 支持的输入格式

| 框架 | 模型格式 | framework 参数值 |
|------|---------|-----------------|
| Caffe | .prototxt + .caffemodel | 0 |
| MindSpore | .mindir | 1 |
| TensorFlow | .pb / .meta / SavedModel | 3 |
| TensorFlow Lite | .tflite | 4 |
| ONNX | .onnx | 5 |

---

## 基本用法

### ONNX 模型转换

```bash
atc --model=resnet50.onnx \
    --framework=5 \
    --output=resnet50 \
    --soc_version=Ascend910B3 \
    --input_shape="input:1,3,224,224" \
    --input_format=NCHW \
    --log=error
```

### TensorFlow 模型转换

```bash
atc --model=model.pb \
    --framework=3 \
    --output=model \
    --soc_version=Ascend910B3 \
    --input_shape="input:1,224,224,3" \
    --input_format=NHWC \
    --output_type=FP16
```

### MindSpore 模型转换

```bash
atc --model=model.mindir \
    --framework=1 \
    --output=model \
    --soc_version=Ascend910B3
```

---

## 关键参数详解

### 基础参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model` | 输入模型路径 | `model.onnx` |
| `--framework` | 框架类型 | `5`（ONNX） |
| `--output` | 输出文件名（不含.om） | `model_output` |
| `--soc_version` | 目标芯片型号 | `Ascend910B3` |
| `--log` | 日志级别 | `info/warning/error` |

### 输入输出参数

```bash
# 静态输入形状
--input_shape="input_name:batch,channel,height,width"

# 多输入
--input_shape="input1:1,3,224,224;input2:1,1000"

# 动态 batch（指定档位）
--dynamic_batch_size="1,2,4,8"

# 动态分辨率（指定档位）
--dynamic_image_size="224,224;448,448;640,640"

# 动态维度（最灵活）
--dynamic_dims="1,3,224,224;2,3,448,448"

# 输入数据格式
--input_format=NCHW  # 或 NHWC, ND

# 输出数据类型
--output_type=FP16   # 或 FP32, INT8
```

### 精度模式参数

```bash
# 精度模式（影响 FP32 算子的处理方式）
--precision_mode=allow_fp32_to_fp16  # 允许 FP32 转 FP16（默认，性能优先）
--precision_mode=force_fp16          # 强制全部使用 FP16
--precision_mode=allow_mix_precision # 混合精度
--precision_mode=must_keep_origin_dtype  # 保持原始精度

# 指定特定算子保持 FP32
--keep_dtype=keep_dtype.cfg
```

`keep_dtype.cfg` 格式：
```
# 保持特定算子为 FP32
op_name:Softmax
op_name:LayerNorm
```

### 优化参数

```bash
# 开启算子融合
--fusion_switch_file=fusion_switch.cfg

# 设置算子编译缓存
--op_compiler_cache_mode=enable
--op_compiler_cache_dir=./op_cache

# 开启 AOE 自动调优
--enable_scope_fusion_passes=true

# 设置内存优化级别
--memory_optimize_level=O1  # 或 O2
```

---

## 动态输入转换

### 动态 Batch

```bash
# 支持 batch=1,2,4,8 四个档位
atc --model=model.onnx \
    --framework=5 \
    --output=model_dynamic_batch \
    --soc_version=Ascend910B3 \
    --input_shape="input:-1,3,224,224" \
    --dynamic_batch_size="1,2,4,8"
```

推理时需要指定实际 batch：
```c
aclmdlSetDynamicBatchSize(modelId, inputDataset, dynInputIdx, 4);
```

### 动态分辨率

```bash
# 支持多种分辨率
atc --model=model.onnx \
    --framework=5 \
    --output=model_dynamic_hw \
    --soc_version=Ascend910B3 \
    --input_shape="input:1,3,-1,-1" \
    --dynamic_image_size="224,224;320,320;416,416;640,640"
```

### 动态维度（最通用）

```bash
# 完全动态的输入形状
atc --model=model.onnx \
    --framework=5 \
    --output=model_dynamic_dims \
    --soc_version=Ascend910B3 \
    --input_shape="input:-1,3,-1,-1" \
    --dynamic_dims="1,3,224,224;2,3,448,448;4,3,640,640"
```

---

## 量化转换

### INT8 量化

```bash
# 使用量化配置文件进行 INT8 量化
atc --model=model.onnx \
    --framework=5 \
    --output=model_int8 \
    --soc_version=Ascend910B3 \
    --quant_config=quant_config.json
```

`quant_config.json` 示例：
```json
{
    "quant_type": "int8",
    "calibration_data": "./calibration_data",
    "calibration_method": "MinMax",
    "skip_layers": ["Softmax", "output"]
}
```

---

## 自定义算子集成

当模型包含自定义算子时，需要在转换时指定：

```bash
atc --model=model.onnx \
    --framework=5 \
    --output=model_custom \
    --soc_version=Ascend910B3 \
    --op_name_map=op_map.cfg \
    --custom_op_lib=./libcustom_op.so
```

`op_map.cfg` 格式（ONNX 算子名 → 昇腾算子名映射）：
```
MyCustomOp:CustomOpImpl
```

---

## 转换结果分析

### 查看 .om 文件信息

```bash
# 使用 omg 工具查看模型信息
omg --model=model.om --mode=1

# 输出示例：
# Model name: resnet50
# Input count: 1
#   Input[0]: name=input, shape=[1,3,224,224], dtype=float16
# Output count: 1
#   Output[0]: name=output, shape=[1,1000], dtype=float16
```

### 转换日志分析

```bash
# 开启详细日志
atc --model=model.onnx \
    --framework=5 \
    --output=model \
    --soc_version=Ascend910B3 \
    --log=info 2>&1 | tee atc_log.txt

# 关键日志信息：
# [INFO] Fusion pass: Conv2D_BN_ReLU applied  ← 融合成功
# [WARNING] Op Softmax: FP32 not supported, converting to FP16  ← 精度转换
# [ERROR] Op CustomOp: not found in op library  ← 算子缺失
```

---

## 常见问题

### 问题1：算子不支持

```
[ERROR] Op type [xxx] is not supported
```

解决方案：
1. 检查 CANN 版本是否支持该算子
2. 使用 `--op_name_map` 映射到等价算子
3. 开发自定义算子

### 问题2：形状推导失败

```
[ERROR] Shape inference failed for node [xxx]
```

解决方案：
1. 检查 `--input_shape` 参数是否正确
2. 确认模型中没有不支持的动态形状操作

### 问题3：精度损失

```
# 推理结果与原始框架差异较大
```

解决方案：
```bash
# 保持原始精度
--precision_mode=must_keep_origin_dtype

# 或指定敏感层保持 FP32
--keep_dtype=sensitive_layers.cfg
```

### 问题4：内存不足

```
[ERROR] Memory allocation failed
```

解决方案：
```bash
# 减小 batch size
--input_shape="input:1,3,224,224"  # 使用 batch=1

# 开启内存优化
--memory_optimize_level=O2
```

---

## ATC 转换脚本模板

```bash
#!/bin/bash
# convert_model.sh

MODEL_NAME="resnet50"
INPUT_MODEL="${MODEL_NAME}.onnx"
OUTPUT_MODEL="${MODEL_NAME}_ascend"
SOC_VERSION="Ascend910B3"
INPUT_SHAPE="input:1,3,224,224"

echo "Converting ${INPUT_MODEL} to ${OUTPUT_MODEL}.om ..."

atc \
    --model=${INPUT_MODEL} \
    --framework=5 \
    --output=${OUTPUT_MODEL} \
    --soc_version=${SOC_VERSION} \
    --input_shape="${INPUT_SHAPE}" \
    --input_format=NCHW \
    --precision_mode=allow_fp32_to_fp16 \
    --op_compiler_cache_mode=enable \
    --op_compiler_cache_dir=./op_cache \
    --log=warning

if [ $? -eq 0 ]; then
    echo "Conversion successful: ${OUTPUT_MODEL}.om"
    ls -lh ${OUTPUT_MODEL}.om
else
    echo "Conversion failed!"
    exit 1
fi
```
