# 特性介绍

SNPE（Snapdragon Neural Processing Engine），是一个在高通骁龙系列芯片上进行神经网络推理的框架，SNPE toolkit包含的主要功能或特性有：

- 端侧网络推理，支持的后端包括骁龙CPU，Adreno GPU以及Hexagon DSP；
- x86 Ubuntu Linux主机端调试，仅支持CPU；
- 多种训练框架格式转换到SNPE DLC（Deep Learning Containe）格式用于模型推理，支持Caffe，Caffe2，ONNX，TensorFlow和PyTorch；
- 模型定点量化功能，主要用于Hexagon DSP相关后端的推理；
- 针对模型推理阶段的分析和调试工具；
- 可通过C++/Java接口，将神经网络推理功能集成到可执行程序或Android APP中。

下图展示了基于SNPE部署网络的workflow：

<div align=center>
<img src="./imgs/7.4.1.jpg" width="600" height="300">
</div>
<div align=center>图1. SNPE workflow </div>

该workflow可以简单概括为以下步骤：首先，选择一个SNPE支持转换模型的框架，经过训练/测试等步骤后，输出待部署的模型；然后，将该模型转换到SNPE推理所需的DLC模式，如果需要在DSP/AIP后端硬件上进行推理，则需要额外对模型进行一次量化，或在推理阶段选择在线量化；最后，在运行时指定预设的后端，给模型输入数据，进行网络推理。

以下是针对SNPE/AIP更详细的介绍：

[Snapdragon NPE Runtime](./subpages/Snapdragon%20NPE%20Runtime.md)

[AIP Runtime](./subpages/AIP%20Runtime.md)