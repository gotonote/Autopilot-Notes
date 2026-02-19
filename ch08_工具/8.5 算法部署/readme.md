# 智驾算法部署实战

## 一、部署概述

智驾算法部署流程：

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  1. 模型训练 │ → │  2. 模型转换  │ → │  3. 推理部署  │
└──────────────┘    └──────────────┘    └──────────────┘
       PyTorch            ONNX              TensorRT
       TensorFlow         TensorFlow-Lite   CUDA/C++
       PaddlePaddle       ONNX-TensorRT     TensorRT Python
                                           TensorRT C++
```

## 二、TensorRT 部署

### 2.1 TensorRT 简介

NVIDIA TensorRT 是高性能深度学习推理引擎：
- 支持 FP32/FP16/INT8/INT4 量化
- CUDA 核心优化
- 层融合与张量内存优化
- 支持 CUDA 动态批量处理

### 2.2 环境安装

```bash
# 安装 TensorRT (Ubuntu)
# 方式1: Tar 文件安装
wget https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.6.1/targz/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
tar -xzf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
export PATH=$PATH:/path/to/TensorRT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/TensorRT/lib

# 方式2: pip 安装
pip install tensorrt

# 验证安装
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

### 2.3 PyTorch 模型转 TensorRT

```python
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# ============================================
# 方式1: 通过 ONNX 转换 (推荐)
# ============================================

# 1. PyTorch 模型导出为 ONNX
class BEVModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 64, 3, padding=1),
            torch.nn.ReLU()
        )
        self.head = torch.nn.Conv2d(64, 10, 1)  # 10 classes
    
    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)

# 导出 ONNX
model = BEVModel()
model.eval()

dummy_input = torch.randn(1, 256, 200, 200)
torch.onnx.export(
    model,
    dummy_input,
    "bev_model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch', 2: 'height', 3: 'width'},
        'output': {0: 'batch'}
    },
    opset_version=13
)
print("ONNX model exported successfully")

# ============================================
# 2. ONNX 转 TensorRT Engine
# ============================================

# 创建 Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file, engine_file):
    """构建 TensorRT 引擎"""
    # 创建 Builder
    builder = trt.Builder(TRT_LOGGER)
    
    # 创建 Network
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    
    # 创建 Parser
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 解析 ONNX
    with open(onnx_file, 'rb') as f:
        parser.parse(f.read())
    
    # 构建配置
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    
    # FP16 加速
    if builder.platform_has_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # INT8 量化 (需要校准数据)
    # config.set_flag(trt.BuilderFlag.INT8)
    
    # 构建引擎
    engine = builder.build_serialized_network(network, config)
    
    # 保存引擎
    with open(engine_file, 'wb') as f:
        f.write(engine)
    
    print(f"TensorRT engine saved to {engine_file}")
    return engine

# 构建引擎
engine = build_engine("bev_model.onnx", "bev_model.trt")

# ============================================
# 3. TensorRT 推理
# ============================================

class TensorRTInference:
    def __init__(self, engine_file):
        # 加载引擎
        with open(engine_file, 'rb') as f:
            engine_bytes = f.read()
        
        # 创建 Runtime
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        
        # 创建 Context
        self.context = self.engine.create_execution_context()
        
        # 分配 GPU 内存
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            
            # 计算内存大小
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            # 分配 CUDA 内存
            cuda_mem = cuda.mem_alloc(size * dtype().itemsize)
            self.bindings.append(int(cuda_mem))
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'name': name, 'shape': shape, 'mem': cuda_mem})
            else:
                self.outputs.append({'name': name, 'shape': shape, 'mem': cuda_mem})
        
        # 创建 Stream
        self.stream = cuda.Stream()
    
    def infer(self, input_data):
        # 拷贝输入数据到 GPU
        cuda.memcpy_htod_async(
            self.inputs[0]['mem'],
            np.ascontiguousarray(input_data),
            self.stream
        )
        
        # 执行推理
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # 拷贝输出数据到 CPU
        output = np.empty(self.outputs[0]['shape'], dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.outputs[0]['mem'], self.stream)
        self.stream.synchronize()
        
        return output

# 使用示例
trt_infer = TensorRTInference("bev_model.trt")

# 准备输入
input_data = np.random.randn(1, 256, 200, 200).astype(np.float32)

# 推理
output = trt_infer.infer(input_data)
print(f"Output shape: {output.shape}")
```

### 2.4 INT8 量化部署

```python
# INT8 量化需要校准数据
class INT8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data_path, cache_file):
        super().__init__()
        self.data_path = calibration_data_path
        self.cache_file = cache_file
        self.batch_size = 8
        self.input_shape = (256, 200, 200)
        
        # 加载校准数据
        self.images = self.load_calibration_data()
        
        # 分配 GPU 内存
        self.d_input = cuda.mem_alloc(
            self.batch_size * np.prod(self.input_shape) * 4
        )
        
        self.data_idx = 0
    
    def load_calibration_data(self):
        # 加载校准图像
        images = []
        data_dir = self.data_path
        
        for img_file in os.listdir(data_dir)[:500]:  # 500张图
            img = cv2.imread(os.path.join(data_dir, img_file))
            img = cv2.resize(img, (200, 200))
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            images.append(img)
        
        return np.array(images, dtype=np.float32) / 255.0
    
    def get_batch(self, names):
        if self.data_idx >= len(self.images):
            self.data_idx = 0
        
        batch = self.images[
            self.data_idx:self.data_idx + self.batch_size
        ]
        self.data_idx += self.batch_size
        
        cuda.memcpy_htod(self.d_input, batch)
        return [int(self.d_input)]
    
    def get_batch_size(self):
        return self.batch_size
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)

# 构建 INT8 引擎
calibrator = INT8Calibrator(
    calibration_data_path='./calibration_data',
    cache_file='./int8_calibration.cache'
)

config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = calibrator

engine = builder.build_serialized_network(network, config)
```

### 2.5 动态尺寸 TensorRT

```python
# 支持动态尺寸输入
def build_dynamic_engine(onnx_file, engine_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_file, 'rb') as f:
        parser.parse(f.read())
    
    # 设置动态形状范围
    profile = builder.create_optimization_profile()
    
    # 获取输入张量名称
    input_tensor_name = network.get_input(0).name
    min_shape = (1, 256, 100, 100)
    opt_shape = (1, 256, 200, 200)
    max_shape = (1, 256, 400, 400)
    
    profile.set_shape(input_tensor_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    
    # 构建引擎
    engine = builder.build_serialized_network(network, config)
    
    with open(engine_file, 'wb') as f:
        f.write(engine)

# 使用动态尺寸推理
def infer_dynamic(trt_engine, input_data):
    context = trt_engine.create_execution_context()
    
    # 设置实际输入形状
    actual_shape = input_data.shape
    context.set_input_shape('input', actual_shape)
    
    # 分配内存
    output_shape = (actual_shape[0], 10, actual_shape[2], actual_shape[3])
    d_input = cuda.mem_alloc(np.prod(actual_shape) * 4)
    d_output = cuda.mem_alloc(np.prod(output_shape) * 4)
    
    # 推理
    cuda.memcpy_htod(d_input, input_data)
    context.execute_v2(bindings=[int(d_input), int(d_output)])
    
    # 获取输出
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output, d_output)
    
    return output
```

## 三、ONNX 部署

### 3.1 ONNX 简介

ONNX (Open Neural Network Exchange) 是开放的神经网络交换格式：
- 跨框架模型转换
- 跨平台部署
- 丰富的算子支持
- 推理运行时优化

### 3.2 模型转换

```python
# ============================================
# PyTorch -> ONNX
# ============================================

import torch
import torch.nn as nn

class SimpleDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.head = nn.Conv2d(128, 6, 1)  # [x, y, w, h, conf, cls]
    
    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

model = SimpleDetector()
model.eval()

# 导出
dummy_input = torch.randn(1, 3, 416, 416)
torch.onnx.export(
    model,
    dummy_input,
    "detector.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch'},
        'output': {0: 'batch'}
    }
)

# ============================================
# TensorFlow -> ONNX
# ============================================

# pip install tf2onnx
import tf2onnx
import tensorflow as tf

# 加载 TensorFlow 模型
model = tf.keras.models.load_model('model.h5')

# 转换为 ONNX
spec = (tf.TensorSpec((None, 416, 416, 3), tf.float32, name='input'),)
model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    output_path='model.onnx'
)

# ============================================
# PaddlePaddle -> ONNX
# ============================================

# pip install paddle2onnx
import paddle2onnx as p2o
import paddle

# 加载模型
model = paddle.jit.load('model')

# 转换为 ONNX
p2o.convert(
    model,
    input_spec=[paddle.static.InputSpec(shape=[1, 3, 416, 416], dtype='float32')],
    opset_version=13,
    output_file='model.onnx'
)
```

### 3.3 ONNX Runtime 推理

```python
# 安装 ONNX Runtime
# pip install onnxruntime-gpu  # GPU 版本
# pip install onnxruntime        # CPU 版本

import onnxruntime as ort
import numpy as np
import cv2

class ONNXInference:
    def __init__(self, model_path, use_gpu=True):
        # 配置 Session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = \
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 选择执行提供程序
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
            if use_gpu else ['CPUExecutionProvider']
        
        # 创建 InferenceSession
        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=providers
        )
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def preprocess(self, image_path):
        """图像预处理"""
        img = cv2.imread(image_path)
        img = cv2.resize(img, (416, 416))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0)
        return img
    
    def infer(self, input_data):
        """推理"""
        output = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )
        return output[0]

# 使用示例
onnx_infer = ONNXInference("detector.onnx", use_gpu=True)

# 预处理
input_data = onnx_infer.preprocess("test.jpg")

# 推理
output = onnx_infer.infer(input_data)
print(f"Output shape: {output.shape}")
```

### 3.4 ONNX 模型优化

```python
# 使用 onnxsim 简化模型
import onnx
from onnxsim import simplify

# 加载模型
model = onnx.load("model.onnx")

# 简化
model_simp, check = simplify(model)

# 保存
onnx.save(model_simp, "model_simp.onnx")

# ============================================
# 使用 onnxruntime-tools 优化
# ============================================
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.float16 import convert_float16_to_float32

# 优化模型
optimized_model = optimizer.optimize_model(
    "model.onnx",
    num_heads=8,
    hidden_size=256,
    optimization_options=optimizer.OptimizationOptions(
        enable_embed_layer_norm=False
    )
)
optimized_model.save_model_to_file("model_optimized.onnx")

# FP16 转换
convert_float16_to_float32(
    input_model='model.onnx',
    output_model='model_fp16.onnx'
)
```

### 3.5 量化部署

```python
# ONNX 动态量化 (PTQ)
import onnx
from onnx.quantization import quantize_dynamic, QuantType

# 动态量化 (无需校准数据)
quantize_dynamic(
    'model.onnx',
    'model_int8.onnx',
    weight_type=QuantType.QInt8,
    optimize_model=True
)

# ============================================
# 静态量化 (需要校准数据)
# ============================================
from onnx.quantization import quantize, QuantType, CalibrationDataReader

class CalibDataReader(CalibrationDataReader):
    def __init__(self, data_list):
        self.data_list = data_list
        self.index = 0
    
    def get_next(self):
        if self.index >= len(self.data_list):
            return None
        self.index += 1
        return {'input': self.data_list[self.index - 1]}

# 准备校准数据
calib_data = [np.random.randn(1, 3, 416, 416).astype(np.float32) 
              for _ in range(100)]
dr = CalibDataReader(calib_data)

# 静态量化
quantize(
    'model.onnx',
    'model_static_quant.onnx',
    quantization_mode=onnx.quantization.QuantizationMode.QLinearOps,
    calibration_data_reader=dr,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8
)
```

## 四、实战案例：BEVFormer 部署

### 4.1 模型导出

```python
# BEVFormer ONNX 导出
import torch
from bevformer import build_model

# 加载模型
model = build_model(config, ckpt='ckpt.pth')
model.eval()

# 准备输入
class DetectedInputs:
    def __init__(self):
        self.img_metas = [{
            'filename': ['test.jpg'],
            'ori_shape': (375, 1242, 3),
            'img_shape': (256, 704, 3),
            'scale_factor': [0.53, 0.53, 0.53, 0.53],
            'pad_shape': (256, 704, 3),
            'crop_shape': [256, 704],
        }]
        
inputs = {
    'img': [torch.randn(1, 6, 3, 256, 704)],  # 6个相机
    'metas': [DetectedInputs()]
}

# 导出
with torch.no_grad():
    torch.onnx.export(
        model,
        (inputs['img'], inputs['metas']),
        'bevformer.onnx',
        input_names=['img', 'metas'],
        output_names=['bev_pe', 'query', 'query_pos', 'img_feats'],
        opset_version=14,
        dynamic_axes={
            'img': {0: 'batch', 1: 'cameras'},
            'bev_pe': {0: 'batch'},
            'query': {0: 'batch'},
        }
    )
```

### 4.2 TensorRT 部署

```python
# BEVFormer TensorRT 推理
class BEVFormerTRT:
    def __init__(self, engine_path):
        self.trt_engine = TensorRTInference(engine_path)
    
    def infer(self, images):
        # images: (B, 6, 3, H, W) 6个相机图像
        B, N, C, H, W = images.shape
        
        # 合并为 (B*N, C, H, W)
        images = images.reshape(B * N, C, H, W)
        
        # 推理
        bev_features = self.trt_engine.infer(images)
        
        # 重塑为 BEV 特征
        bev_features = bev_features.reshape(B, N, -1, H//4, W//4)
        
        return bev_features

# 使用
bev_trt = BEVFormerTRT('bevformer.trt')
images = np.random.randn(1, 6, 3, 256, 704).astype(np.float32)
bev_features = bev_trt.infer(images)
print(f"BEV features shape: {bev_features.shape}")
```

## 五、性能优化技巧

### 5.1 CUDA 优化

```python
# 批量推理
def batch_infer(trt_engine, images, batch_size=8):
    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        result = trt_engine.infer(batch)
        results.append(result)
    return np.concatenate(results, axis=0)

# CUDA Stream 并行
streams = [cuda.Stream() for _ in range(4)]
for i, stream in enumerate(streams):
    # 异步推理
    cuda.memcpy_htod_async(d_input, batch[i], stream)
    context.execute_async_v2(bindings, stream_handle=stream.handle)

# 等待所有流完成
for stream in streams:
    stream.synchronize()
```

### 5.2 内存优化

```python
# 内存池复用
class MemoryPool:
    def __init__(self, size, dtype):
        self.size = size
        self.dtype = dtype
        self.pool = []
    
    def allocate(self):
        if self.pool:
            return self.pool.pop()
        return cuda.mem_alloc(self.size * np.dtype(self.dtype).itemsize)
    
    def release(self, mem):
        self.pool.append(mem)

# 使用内存池
mem_pool = MemoryPool(1024 * 1024 * 1024, np.float32)
d_input = mem_pool.allocate()
# ... 推理
mem_pool.release(d_input)
```

## 六、部署 checklist

| 项目 | 内容 |
|------|------|
| ✅ 模型检查 | ONNX 算子兼容性验证 |
| ✅ 精度验证 | 对比 PyTorch 输出误差 < 1% |
| ✅ 性能测试 | 延迟/吞吐满足实时要求 |
| ✅ 量化校准 | INT8 精度损失 < 2% |
| ✅ 内存优化 | GPU 显存占用合理 |
| ✅ 异常处理 | 边界输入鲁棒性 |
