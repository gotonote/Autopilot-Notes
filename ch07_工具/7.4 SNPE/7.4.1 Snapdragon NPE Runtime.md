# **Snapdragon NPE Runtime**

**Overview**

以下是SNPE软硬件架构的总览图，图中详细介绍了SNPE运行时涉及到的软件和硬件层，包括Runtime Library和常用后端，以及一些简单的硬件实现。

![https://developer.qualcomm.com/sites/default/files/docs/snpe/images/snpe_runtime.png](https://developer.qualcomm.com/sites/default/files/docs/snpe/images/snpe_runtime.png)

软件层功能主要包括 :

**DL Container Loader :** 主要涉及从训练框架到DLC目标格式的转换器，该转换器部分由python实现，可以修改模型转换的逻辑；

**Model Validation :** 校验模型是否支持，具体可以查看 [SNPE Layer 支持列表](https://developer.qualcomm.com/sites/default/files/docs/snpe/network_layers.html#network_layers)，该列表在某些版本中与转换器的支持列表有出入，可以查看python代码确认；

**Runtime Engine :** 推理引擎，其中内置了Profiling功能，可以在网络初始化时打开，输出的报告可以通过自带的解析器进行解析；UDL或UDO功能，是官方提供的扩展SNPE推理支持的方式，类似于TensorRT中Plugin的概念，可以基于Hexagon SDK进行开发；

**Partitioning Logic :** 这是一个比较复杂的逻辑，可以简单将它理解为一个“回退”机制：在网络初始化阶段，SNPE允许输入一个backend列表，按照列表中的优先级顺序从高到低地选择网络跑在哪一个后端，当第一优先级后端对网络中某个层不支持的时候，SNPE会尝试将这个层回退到更低优先级的后端。这种做法增加了网络的支持范围，但引入了性能的损耗；

硬件层或硬件后端主要包括：

**CPU Runtime**: 使网络运行在骁龙CPU上，支持FP32/INT8精度模式；

**GPU Runtime**: 使网络运行在Adreno GPU上，支持FP32/FP16以及a32w16的混合精度模式；

**DSP Runtime**: 使网络运行在Hexagon DSP上，支持INT8精度模式；图中的Q6是一个软件概念，对应DSP上的神经网络实现，HVX对应硬件的向量计算单元；

**AIP Runtime**: 使网络运行在Hexagon DSP和CPU的混合模式上，官方称为HTA，支持INT8或a16w8精度模式，需要注意的是，HTA仅在特定硬件上存在，例如：SP855/SP865/XR2。