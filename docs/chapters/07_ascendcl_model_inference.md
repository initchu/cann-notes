# 模型推理与 DVPP 媒体处理

## 模型管理 API

### 模型加载方式

CANN 支持三种模型加载方式：

```c
// 方式一：从文件加载（最常用）
uint32_t modelId;
aclmdlLoadFromFile("model.om", &modelId);

// 方式二：从内存加载（适合嵌入式场景）
void* modelData;
size_t modelSize;
// ... 读取 .om 文件到 modelData ...
aclmdlLoadFromMem(modelData, modelSize, &modelId);

// 方式三：从文件加载，指定内存（精细控制）
void* workPtr;   // 模型运行时工作内存
size_t workSize;
void* weightPtr; // 模型权重内存
size_t weightSize;
aclmdlQuerySize("model.om", &workSize, &weightSize);
aclrtMalloc(&workPtr, workSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&weightPtr, weightSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclmdlLoadFromFileWithMem("model.om", &modelId, workPtr, workSize, 
                           weightPtr, weightSize);
```

### 模型描述查询

```c
aclmdlDesc* modelDesc = aclmdlCreateDesc();
aclmdlGetDesc(modelDesc, modelId);

// 查询输入信息
size_t inputCount = aclmdlGetNumInputs(modelDesc);
for (size_t i = 0; i < inputCount; i++) {
    // 输入大小
    size_t inputSize = aclmdlGetInputSizeByIndex(modelDesc, i);
    
    // 输入名称
    const char* inputName = aclmdlGetInputNameByIndex(modelDesc, i);
    
    // 输入维度
    aclmdlIODims dims;
    aclmdlGetInputDims(modelDesc, i, &dims);
    printf("Input[%zu]: name=%s, size=%zu, dims=[", i, inputName, inputSize);
    for (size_t d = 0; d < dims.dimCount; d++) {
        printf("%ld%s", dims.dims[d], d < dims.dimCount-1 ? "," : "");
    }
    printf("]\n");
    
    // 数据类型和格式
    aclDataType dataType = aclmdlGetInputDataType(modelDesc, i);
    aclFormat format = aclmdlGetInputFormat(modelDesc, i);
}

// 查询输出信息（类似输入）
size_t outputCount = aclmdlGetNumOutputs(modelDesc);
for (size_t i = 0; i < outputCount; i++) {
    size_t outputSize = aclmdlGetOutputSizeByIndex(modelDesc, i);
    const char* outputName = aclmdlGetOutputNameByIndex(modelDesc, i);
    // ...
}
```

---

## 动态输入推理

### 动态 Batch

```c
// 查询模型支持的动态 batch 档位
aclmdlBatch batchInfo;
aclmdlGetDynamicBatch(modelDesc, &batchInfo);
printf("Supported batch sizes: ");
for (size_t i = 0; i < batchInfo.batchCount; i++) {
    printf("%lu ", batchInfo.batch[i]);
}

// 推理时设置实际 batch 大小
uint64_t actualBatch = 4;
size_t dynamicInputIndex;
aclmdlGetInputIndexByName(modelDesc, ACL_DYNAMIC_TENSOR_NAME, &dynamicInputIndex);
aclmdlSetDynamicBatchSize(modelId, inputDataset, dynamicInputIndex, actualBatch);
```

### 动态分辨率

```c
// 查询支持的分辨率档位
aclmdlHW hwInfo;
aclmdlGetDynamicHW(modelDesc, 0, &hwInfo);
for (size_t i = 0; i < hwInfo.hwCount; i++) {
    printf("Supported HW: %lux%lu\n", hwInfo.hw[i][0], hwInfo.hw[i][1]);
}

// 设置实际分辨率
size_t hwInputIndex;
aclmdlGetInputIndexByName(modelDesc, ACL_DYNAMIC_TENSOR_NAME, &hwInputIndex);
aclmdlSetDynamicHWSize(modelId, inputDataset, hwInputIndex, 480, 640);
```

### 动态维度（最灵活）

```c
// 查询动态维度信息
aclmdlIODims curDims;
aclmdlGetInputDynamicDims(modelDesc, 0, &curDims, 0);

// 设置实际维度
aclmdlIODims actualDims;
actualDims.dimCount = 4;
actualDims.dims[0] = 2;    // batch=2
actualDims.dims[1] = 3;    // channel=3
actualDims.dims[2] = 320;  // height=320
actualDims.dims[3] = 320;  // width=320

size_t dynInputIdx;
aclmdlGetInputIndexByName(modelDesc, ACL_DYNAMIC_TENSOR_NAME, &dynInputIdx);
aclmdlSetInputDynamicDims(modelId, inputDataset, dynInputIdx, &actualDims);
```

---

## DVPP 媒体数据处理

DVPP（Digital Vision Pre-Processing）是昇腾处理器内置的硬件图像/视频处理单元，通过 AscendCL 的 `acldvpp` 接口访问。

### DVPP 初始化

```c
// 创建 DVPP 通道描述符
acldvppChannelDesc* dvppChannelDesc = acldvppCreateChannelDesc();

// 创建 DVPP 通道
acldvppCreateChannel(dvppChannelDesc);
```

### 图像解码（JPEGD）

```c
// 读取 JPEG 文件
FILE* fp = fopen("input.jpg", "rb");
fseek(fp, 0, SEEK_END);
size_t jpegSize = ftell(fp);
rewind(fp);

// 分配 Host 内存存储 JPEG 数据
void* jpegHostPtr;
aclrtMallocHost(&jpegHostPtr, jpegSize);
fread(jpegHostPtr, 1, jpegSize, fp);
fclose(fp);

// 拷贝到 Device
void* jpegDevPtr;
aclrtMalloc(&jpegDevPtr, jpegSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(jpegDevPtr, jpegSize, jpegHostPtr, jpegSize, 
             ACL_MEMCPY_HOST_TO_DEVICE);

// 创建输入图像描述符
acldvppPicDesc* inputPicDesc = acldvppCreatePicDesc();
acldvppSetPicDescData(inputPicDesc, jpegDevPtr);
acldvppSetPicDescSize(inputPicDesc, jpegSize);

// 创建输出图像描述符（解码后的 YUV420SP 格式）
uint32_t outWidth = 224, outHeight = 224;
uint32_t outWidthStride = (outWidth + 127) & ~127;  // 128 字节对齐
uint32_t outHeightStride = (outHeight + 15) & ~15;  // 16 字节对齐
size_t outSize = outWidthStride * outHeightStride * 3 / 2;  // YUV420SP

void* outDevPtr;
aclrtMalloc(&outDevPtr, outSize, ACL_MEM_MALLOC_HUGE_FIRST);

acldvppPicDesc* outputPicDesc = acldvppCreatePicDesc();
acldvppSetPicDescData(outputPicDesc, outDevPtr);
acldvppSetPicDescFormat(outputPicDesc, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
acldvppSetPicDescWidth(outputPicDesc, outWidth);
acldvppSetPicDescHeight(outputPicDesc, outHeight);
acldvppSetPicDescWidthStride(outputPicDesc, outWidthStride);
acldvppSetPicDescHeightStride(outputPicDesc, outHeightStride);
acldvppSetPicDescSize(outputPicDesc, outSize);

// 执行 JPEG 解码（异步）
acldvppJpegDecodeAsync(dvppChannelDesc, jpegDevPtr, jpegSize, 
                        outputPicDesc, stream);
aclrtSynchronizeStream(stream);
```

### 图像缩放（VPC Resize）

```c
// 创建缩放配置
acldvppResizeConfig* resizeConfig = acldvppCreateResizeConfig();

// 执行缩放（YUV420SP → YUV420SP）
acldvppVpcResizeAsync(dvppChannelDesc, 
                       inputPicDesc,   // 输入（解码后的图像）
                       outputPicDesc,  // 输出（缩放后的图像）
                       resizeConfig,   // 缩放配置
                       stream);
aclrtSynchronizeStream(stream);

acldvppDestroyResizeConfig(resizeConfig);
```

### 图像裁剪（VPC Crop）

```c
// 定义裁剪区域
acldvppRoiConfig* cropArea = acldvppCreateRoiConfig(
    0, 0,       // 左上角 (x, y)
    224, 224    // 右下角 (x, y)
);

// 执行裁剪
acldvppVpcCropAsync(dvppChannelDesc, inputPicDesc, outputPicDesc, 
                     cropArea, stream);
aclrtSynchronizeStream(stream);

acldvppDestroyRoiConfig(cropArea);
```

### 色彩空间转换（YUV → RGB）

```c
// DVPP 输出通常是 YUV420SP 格式
// 需要转换为 RGB 才能送入模型（或在 ATC 转换时处理）

// 方式一：使用 VPC 色彩转换
// 方式二：在 ATC 模型转换时插入色彩转换算子（推荐）
// 方式三：使用自定义算子
```

### 完整图像预处理流水线

```
JPEG 文件
    ↓ JPEGD（硬件解码）
YUV420SP 原始图像
    ↓ VPC Resize（硬件缩放）
YUV420SP 224×224
    ↓ VPC 色彩转换（可选）
RGB 224×224
    ↓ 归一化（算子）
FP16 Tensor [1,3,224,224]
    ↓ 模型推理
输出结果
```

---

## 视频推理流水线

```c
// 视频解码 + 推理的典型流水线
acldvppStreamDesc* streamDesc = acldvppCreateStreamDesc();

// 设置视频流参数
acldvppSetStreamDescData(streamDesc, videoData, videoSize);
acldvppSetStreamDescFormat(streamDesc, H265_MAIN_LEVEL);

// 创建视频解码通道
acldvppChannelDesc* vdecChannelDesc = acldvppCreateChannelDesc();
acldvppSetChannelDescMode(vdecChannelDesc, DVPP_CHNMODE_VB);

// 注册回调函数（每帧解码完成后调用）
acldvppVdecSetChannelDescCallback(vdecChannelDesc, vdecCallback);

// 创建解码通道
acldvppVdecCreateChannel(vdecChannelDesc);

// 发送视频帧进行解码
acldvppVdecSendFrame(vdecChannelDesc, streamDesc, outputPicDesc, 
                      nullptr, stream);
```

---

## 推理性能优化

### 1. 批处理（Batching）

```c
// 将多个样本合并为一个 batch 推理
// batch=8 比 8 次 batch=1 推理效率高得多
// 需要在 ATC 转换时指定支持的 batch 档位
```

### 2. 流水线并行

```
时间轴：
T1: [预处理1] [推理1]         [后处理1]
T2:           [预处理2] [推理2]         [后处理2]
T3:                    [预处理3] [推理3]         [后处理3]
```

### 3. 多模型并发

```c
// 在不同 Stream 上并发执行多个模型
aclmdlExecuteAsync(model1Id, input1, output1, stream1);
aclmdlExecuteAsync(model2Id, input2, output2, stream2);

// 等待两个模型都完成
aclrtSynchronizeStream(stream1);
aclrtSynchronizeStream(stream2);
```

### 4. 模型预热

```c
// 首次推理会触发 JIT 编译，耗时较长
// 建议在正式推理前进行预热
for (int i = 0; i < 3; i++) {
    aclmdlExecute(modelId, warmupInput, warmupOutput);
}
// 之后的推理延迟才是真实性能
```
