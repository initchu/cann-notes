# 昇腾应用部署实践

## 部署场景概览

昇腾平台支持多种部署场景：

| 场景 | 硬件 | 典型产品 | 特点 |
|------|------|---------|------|
| 云端推理 | Ascend 310P/910 | Atlas 300I | 高吞吐，低延迟 |
| 边缘推理 | Ascend 310 | Atlas 200 DK | 低功耗，小体积 |
| 端侧推理 | 麒麟 NPU | 华为手机 | 极低功耗 |
| 云端训练 | Ascend 910B/C | Atlas 800T A2 | 超高算力 |

---

## 推理服务部署

### 方式一：直接使用 AscendCL

适合对延迟要求极高的场景，直接调用 AscendCL API：

```cpp
// 高性能推理服务核心代码
class InferenceEngine {
public:
    bool Init(const std::string& modelPath, int deviceId) {
        aclInit(nullptr);
        aclrtSetDevice(deviceId);
        aclrtCreateContext(&context_, deviceId);
        aclrtCreateStream(&stream_);
        
        aclmdlLoadFromFile(modelPath.c_str(), &modelId_);
        modelDesc_ = aclmdlCreateDesc();
        aclmdlGetDesc(modelDesc_, modelId_);
        
        // 预分配输入输出内存
        PrepareBuffers();
        return true;
    }
    
    std::vector<float> Infer(const std::vector<float>& input) {
        // 上传输入
        aclrtMemcpy(inputDevPtr_, inputSize_, 
                    input.data(), inputSize_,
                    ACL_MEMCPY_HOST_TO_DEVICE);
        
        // 执行推理
        aclmdlExecute(modelId_, inputDataset_, outputDataset_);
        
        // 下载输出
        std::vector<float> output(outputSize_ / sizeof(float));
        aclrtMemcpy(output.data(), outputSize_,
                    outputDevPtr_, outputSize_,
                    ACL_MEMCPY_DEVICE_TO_HOST);
        return output;
    }
    
private:
    uint32_t modelId_;
    aclrtContext context_;
    aclrtStream stream_;
    aclmdlDesc* modelDesc_;
    aclmdlDataset* inputDataset_;
    aclmdlDataset* outputDataset_;
    void* inputDevPtr_;
    void* outputDevPtr_;
    size_t inputSize_, outputSize_;
};
```

### 方式二：使用 Triton Inference Server

```bash
# 安装 Triton 昇腾后端
pip install tritonclient[all]

# 模型仓库结构
model_repository/
└── resnet50/
    ├── config.pbtxt
    └── 1/
        └── model.om
```

`config.pbtxt` 配置：
```protobuf
name: "resnet50"
backend: "ascend"
max_batch_size: 8

input [
  {
    name: "input"
    data_type: TYPE_FP16
    dims: [3, 224, 224]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP16
    dims: [1000]
  }
]

dynamic_batching {
  preferred_batch_size: [1, 2, 4, 8]
  max_queue_delay_microseconds: 100
}
```

### 方式三：使用 MindSpore Serving

```python
# 服务端
from mindspore_serving import server

def start_serving():
    server.start_grpc_server("0.0.0.0:5500")
    server.start_restful_server("0.0.0.0:1500")
    
    # 加载模型
    server.start_servables(
        servable_directory="./model_dir",
        servable_name="resnet50",
        device_type="Ascend",
        device_id=0
    )

# 客户端
from mindspore_serving.client import Client

client = Client("localhost:5500", "resnet50", "predict")
result = client.infer({"input": input_data})
```

---

## 容器化部署

### Docker 镜像

```dockerfile
# Dockerfile
FROM ubuntu:20.04

# 安装依赖
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libgomp1 libstdc++6

# 复制 CANN 运行时（NNRT 包）
COPY ascend-cann-nnrt_8.x_linux-aarch64.run /tmp/
RUN /tmp/ascend-cann-nnrt_8.x_linux-aarch64.run --install \
    --install-path=/usr/local/Ascend

# 设置环境变量
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/nnrt/latest
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:$LD_LIBRARY_PATH

# 复制应用
COPY app/ /app/
WORKDIR /app

# 复制模型
COPY model.om /app/model/

CMD ["python3", "server.py"]
```

```bash
# 构建镜像
docker build -t inference-service:latest .

# 运行容器（挂载 NPU 设备）
docker run -d \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -p 8080:8080 \
    inference-service:latest
```

### Kubernetes 部署

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-service
spec:
  replicas: 4
  selector:
    matchLabels:
      app: inference-service
  template:
    metadata:
      labels:
        app: inference-service
    spec:
      containers:
      - name: inference
        image: inference-service:latest
        resources:
          limits:
            huawei.com/Ascend910: 1  # 申请 1 个 Ascend 910
          requests:
            huawei.com/Ascend910: 1
        env:
        - name: ASCEND_DEVICE_ID
          value: "0"
        ports:
        - containerPort: 8080
```

---

## 性能优化部署策略

### 1. 批处理优化

```python
# 动态批处理服务
import asyncio
from collections import deque

class BatchInferenceService:
    def __init__(self, model, max_batch=8, max_wait_ms=10):
        self.model = model
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self.queue = deque()
        self.lock = asyncio.Lock()
    
    async def infer(self, input_data):
        future = asyncio.Future()
        async with self.lock:
            self.queue.append((input_data, future))
        
        # 等待批处理结果
        return await future
    
    async def batch_worker(self):
        while True:
            await asyncio.sleep(self.max_wait_ms / 1000)
            
            async with self.lock:
                if not self.queue:
                    continue
                
                # 收集一批请求
                batch_size = min(len(self.queue), self.max_batch)
                batch_items = [self.queue.popleft() for _ in range(batch_size)]
            
            # 批量推理
            inputs = [item[0] for item in batch_items]
            futures = [item[1] for item in batch_items]
            
            results = self.model.batch_infer(inputs)
            
            # 返回结果
            for future, result in zip(futures, results):
                future.set_result(result)
```

### 2. 多实例部署

```python
# 多 NPU 负载均衡
import threading
from queue import Queue

class MultiNPUInferencePool:
    def __init__(self, model_path, num_devices=8):
        self.engines = []
        self.request_queue = Queue()
        
        for device_id in range(num_devices):
            engine = InferenceEngine()
            engine.Init(model_path, device_id)
            self.engines.append(engine)
            
            # 每个设备一个工作线程
            t = threading.Thread(
                target=self._worker, 
                args=(engine,),
                daemon=True
            )
            t.start()
    
    def _worker(self, engine):
        while True:
            input_data, result_queue = self.request_queue.get()
            result = engine.Infer(input_data)
            result_queue.put(result)
    
    def infer(self, input_data):
        result_queue = Queue()
        self.request_queue.put((input_data, result_queue))
        return result_queue.get()
```

---

## 边缘部署：Atlas 200 DK

Atlas 200 DK 是面向开发者的边缘推理套件：

```bash
# 在 Atlas 200 DK 上部署
# 1. 通过 USB 或网络连接设备
ssh HwHiAiUser@192.168.1.2

# 2. 上传模型和应用
scp model.om HwHiAiUser@192.168.1.2:/home/HwHiAiUser/

# 3. 运行推理
./inference_app --model=model.om --input=test.jpg

# 4. 查看设备状态
npu-smi info
```

### 边缘部署注意事项

- Atlas 200 DK 使用 Ascend 310，算力相对有限
- 内存仅 8GB，需要控制模型大小
- 支持 JPEG/PNG 解码和视频流处理
- 功耗约 8W，适合嵌入式场景

---

## 监控与运维

```python
# 生产环境监控
import subprocess
import json
import time

def get_npu_metrics(device_id=0):
    result = subprocess.run(
        ["npu-smi", "info", "-t", "usages", "-i", str(device_id)],
        capture_output=True, text=True
    )
    # 解析输出...
    return {
        "aicore_utilization": 87,
        "memory_used_mb": 45312,
        "memory_total_mb": 65536,
        "temperature_c": 52,
        "power_w": 285
    }

# 上报到监控系统（Prometheus/Grafana）
from prometheus_client import Gauge, start_http_server

npu_util = Gauge('npu_aicore_utilization', 'AI Core utilization', ['device'])
npu_mem = Gauge('npu_memory_used_bytes', 'Memory used', ['device'])

start_http_server(9090)

while True:
    metrics = get_npu_metrics(0)
    npu_util.labels(device='0').set(metrics['aicore_utilization'])
    npu_mem.labels(device='0').set(metrics['memory_used_mb'] * 1024 * 1024)
    time.sleep(10)
```
