# PreScan 仿真入门指南

## 一、平台简介

PreScan 是西门子旗下的自动驾驶仿真平台，主要特点：
- 基于 MATLAB/Simulink 集成
- 强大的传感器仿真（相机、LiDAR、RADAR、超声波）
- 快速原型开发
- 与 MATLAB/Simulink 深度集成
- 支持 ASAM OpenSCENARIO/OpenDRIVE

> 官网：https://www.siemens.com/software-products/prescan

## 二、系统要求

### 2.1 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | Intel i7 / AMD Ryzen 7 | Intel i9 / AMD Ryzen 9 |
| RAM | 16 GB | 32 GB+ |
| GPU | NVIDIA GTX 1060 | NVIDIA RTX 3080+ |
| 存储 | 50 GB SSD | 100 GB SSD |

### 2.2 软件要求

- Windows 10/11 (64-bit)
- MATLAB R2021b+
- Visual Studio 2019+
- CUDA 11.x+

## 三、安装指南

### 3.1 安装步骤

1. **下载 PreScan**
   - 访问 Siemens Support
   - 下载 PreScan 安装包

2. **运行安装程序**
   ```
   安装路径: C:\Program Files\Siemens\PreScan
   ```

3. **配置 MATLAB**
   - 设置 MATLAB 路径
   - 验证 PreScan 工具箱

4. **激活许可**
   ```
   prescan_lictool.exe -install <license_file>
   ```

### 3.2 首次运行

```bash
# 启动 PreScan GUI
# 方式1: 开始菜单
Start > Siemens > PreScan > PreScan GUI

# 方式2: MATLAB 命令行
matlab
>> prescan
```

## 四、界面介绍

### 4.1 主界面

```
┌─────────────────────────────────────────────────────────┐
│ 菜单栏 │ 文件  编辑  视图  场景  仿真  工具  帮助     │
├─────────────────────────────────────────────────────────┤
│ 工具栏 │ 新建 │ 打开 │ 保存 │ 运行 │ 停止 │ 设置    │
├────────────┬────────────────────────────────────────────┤
│            │                                            │
│   场景树   │           3D 视口预览                        │
│   (左侧)   │                                            │
│            │                                            │
│ - 道路     │                                            │
│ - 车辆     │                                            │
│ - 传感器   │                                            │
│ - 目标物   │                                            │
│            │                                            │
├────────────┴────────────────────────────────────────────┤
│                    状态栏                                │
└─────────────────────────────────────────────────────────┘
```

### 4.2 工作流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  1. 构建场景  │ → │ 2. 配置传感器 │ → │ 3. 添加算法  │
└─────────────┘    └─────────────┘    └─────────────┘
                                               ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 6. 分析结果  │ ← │ 5. 运行仿真  │ ← │ 4. 设置仿真  │
└─────────────┘    └─────────────┘    └─────────────┘
```

## 五、场景构建

### 5.1 道路建模

```matlab
% MATLAB/Simulink 中创建道路
% 方式1: 使用道路编辑器 GUI
prescan.explorer.createRoad

% 方式2: 编程方式创建
roadParams = PreScan.Assets.Road;
roadParams.Name = 'Highway';
roadParams.Length = 500;  % 米
roadParams.Lanes = 3;
roadParams.LaneWidth = 3.5;
road = PreScan.Assets.Road(roadParams);
```

### 5.2 交通场景

```matlab
% 添加车辆
vehicle = PreScan.Assets.Vehicle;
vehicle.Name = 'EgoVehicle';
vehicle.Position = [0, 0, 0];  % x, y, z (米)
vehicle.Orientation = [0, 0, 0];  % roll, pitch, yaw (度)
vehicle.Velocity = 50 / 3.6;  % m/s

% 添加交通参与者
trafficVehicle = PreScan.Assets.Vehicle;
trafficVehicle.Name = 'LeadVehicle';
trafficVehicle.Position = [50, 0, 0];
trafficVehicle.Velocity = 40 / 3.6;
```

### 5.3 目标物

```matlab
% 添加行人
pedestrian = PreScan.Assets.Pedestrian;
pedestrian.Name = 'Pedestrian_1';
pedestrian.Position = [20, 5, 0];
pedestrian.MotionModel = 'Walker';  % 行人行走模型

% 添加自行车
bicycle = PreScan.Assets.Bicycle;
bicycle.Name = 'Bicycle_1';
bicycle.Position = [30, -3, 0];
```

## 六、传感器配置

### 6.1 相机

```matlab
% 添加相机传感器
camera = PreScan.Sensors.Camera;
camera.Name = 'FrontCamera';
camera.Parent = 'EgoVehicle';
camera.Position = [1.5, 0, 1.0];  % 车前1.5米，升高1米
camera.Orientation = [0, 0, 0];

% 相机参数
camera.ImageSize = [1920, 1080];
camera.HorizontalFOV = 120;  % 水平视场角
camera.VerticalFOV = 60;
camera.FocalLength = 4.5;  % mm
```

### 6.2 LiDAR

```matlab
% 添加激光雷达
lidar = PreScan.Sensors.LaserScanner;
lidar.Name = 'FrontLidar';
lidar.Parent = 'EgoVehicle';
lidar.Position = [0, 0, 1.8];

% LiDAR 参数
lidar.Model = 'Velodyne HDL-64E';
lidar.Range = 100;  % 米
lidar.VerticalFOV = [-2, 2];  % 垂直视场角
lidar.NumLasers = 64;
```

### 6.3 RADAR

```matlab
% 添加毫米波雷达
radar = PreScan.Sensors.Radar;
radar.Name = 'FrontRadar';
radar.Parent = 'EgoVehicle';
radar.Position = [0.5, 0, 0.5];

% RADAR 参数
radar.Model = '77GHz Long Range';
radar.Range = 200;
radar.HorizontalFOV = 60;
radar.VerticalFOV = 20;
radar.RangeResolution = 0.5;  % 米
radar.AngularResolution = 2;  % 度
```

### 6.4 超声波

```matlab
% 添加超声波传感器
ultrasonic = PreScan.Sensors.Ultrasonic;
ultrasonic.Name = 'ParkingSensors';
ultrasonic.Parent = 'EgoVehicle';

% 配置多个探头位置
ultrasonic.MountingPositions = [
    0.1,  0.8, 0.3;   % 前左
    0.1, -0.8, 0.3;   % 前右
   -0.1,  0.8, 0.3;   % 后左
   -0.1, -0.8, 0.3    % 后右
];
```

## 七、Simulink 集成

### 7.1 创建模型

```matlab
% 创建 Simulink 模型
modelName = 'MyPreScanDemo';
new_system(modelName, 'Model');
open_system(modelName);

% 添加 PreScan 传感器块
add_block('prescan/Sensors/Camera', [modelName '/Camera']);
add_block('prescan/Sensors/LaserScanner', [modelName '/Lidar']);
add_block('prescan/Sensors/Radar', [modelName '/Radar']);
```

### 7.2 传感器数据处理

```matlab
% 相机数据处理示例
function rgb_image = processCamera(cameraData)
    % cameraData: 相机原始数据
    % 转换为 RGB 图像
    rgb_image = reshape(cameraData.RGB, ...
        cameraData.Height, cameraData.Width, 3);
    
    % 转换为 BGR (OpenCV 格式)
    rgb_image = rgb_image(:, :, [3 2 1]);
end

% LiDAR 点云处理
function pointCloud = processLidar(lidarData)
    % 提取点云 XYZ
    pointCloud.X = lidarData.X;
    pointCloud.Y = lidarData.Y;
    pointCloud.Z = lidarData.Z;
    pointCloud.Intensity = lidarData.Intensity;
end
```

### 7.3 车辆控制

```matlab
% 车辆控制输入块
add_block('prescan/Vehicle/VehicleControl', ...
    [modelName '/VehicleControl']);

% 设置控制信号
% Input: [steer, throttle, brake]
% steer: -1 ~ 1 (左 ~ 右)
% throttle: 0 ~ 1
% brake: 0 ~ 1
```

## 八、仿真运行

### 8.1 配置仿真参数

```matlab
% 设置仿真参数
simParams = prescan.SimulationParameters;
simParams.SimulationTime = 30;  % 仿真时长(秒)
simParams.SampleTime = 0.01;    % 采样时间(秒)
simParams.SolverType = 'FixedStep';  % 求解器类型
simParams.FixedStepSize = 0.01;      % 固定步长
```

### 8.2 运行仿真

```matlab
% 运行仿真
simOut = sim(modelName, 'SimulationMode', 'normal');

% 或使用快速仿真模式
simOut = sim(modelName, 'SimulationMode', 'accelerator');
```

### 8.3 实时仿真

```matlab
% 部署到实时仿真机
simOut = sim(modelName, ...
    'SimulationMode', 'external', ...
    'ExternalInput', inputData);
```

## 九、数据分析

### 9.1 导出数据

```matlab
% 导出仿真数据
logData = simOut.get('logsout');

% 获取车辆轨迹
trajectory = logData.get('EgoVehicle.Position').Values;
time = trajectory.Time;
x = trajectory.Data(:, 1);
y = trajectory.Data(:, 2);

% 绘制轨迹
figure;
plot(x, y);
xlabel('X (m)');
ylabel('Y (m)');
title('Ego Vehicle Trajectory');
grid on;
```

### 9.2 可视化

```matlab
% 使用 PreScan Viewer 查看结果
prescan.postViewer(simOut);

% 导出为 CSV
writematrix([time, x, y], 'trajectory.csv');
```

## 十、与其他工具集成

### 10.1 OpenDRIVE 导入

```matlab
% 导入 OpenDRIVE 地图
road = PreScan.Import.OpenDRIVE('highway.xodr');

% 转换为 PreScan 道路
prescan.assets.road.convertFromOpenDRIVE(road);
```

### 10.2 OpenSCENARIO 导入

```matlab
% 导入场景
scenario = PreScan.Import.OpenSCENARIO('cut_in.xml');

% 解析场景参数
egoVehicle = scenario.getEgoVehicle();
trafficVehicles = scenario.getTrafficVehicles();
```

### 10.3 ROS 集成

```matlab
% 配置 ROS 通信
rosInterface = PreScan.ROS.Interface;
rosInterface.NodeName = '/prescan_node';
rosInterface.Publishers = {
    'EgoVehiclePose', '/vehicle_pose', 'nav_msgs/Odometry'
};
rosInterface.Subscribers = {
    '/control_cmd', '/vehicle_cmd', 'geometry_msgs/Twist'
};
```

## 十一、常见问题

### 11.1 安装问题

```matlab
% MATLAB 路径未识别
prescan.setup.addPath()

% 许可证问题
prescan_lictool.exe -status
```

### 11.2 仿真性能

```matlab
% 降低传感器更新频率
camera.SamplingTime = 0.05;  % 20 Hz

% 减少点云数量
lidar.PointsPerScan = 10000;
```

### 11.3 数据导出

```matlab
% 导出为 HDF5 格式
prescan.export.toHDF5(simOut, 'output.h5');

% 导出为 ROS Bag
prescan.export.toROSBag(simOut, 'output.bag');
```

## 十二、学习资源

- [PreScan 官方文档](https://www.siemens.com/software-products/prescan)
- [PreScan Tutorials](https://www.siemens.com/tutorial-precan)
- [MATLAB Central](https://www.mathworks.com/matlabcentral/)
- [ASAM OpenSCENARIO](https://www.asam.net/standards/detail/osc/)
