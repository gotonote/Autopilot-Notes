# CARLA 仿真完全指南

## 一、平台简介

CARLA 是开源的自动驾驶仿真平台，支持：
- 多元城市场景构建
- 真实物理引擎
- 多种传感器模拟（RGB相机、LiDAR、RADAR、GPS、IMU等）
- 天气和时间模拟
- 交通场景仿真
- Python/C++ API

> 官网：https://carla.org/

## 二、安装指南

### 2.1 Ubuntu 20.04/22.04 安装

```bash
# 安装依赖
sudo apt-get update
sudo apt-get install wget software-properties-common

# 安装 CARLA
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_Latest-release.tar.gz
tar -xzf CARLA_Latest-release.tar.gz
cd CARLA_*/ # 进入解压目录

# 启动仿真器
./CarlaUE4.sh
```

### 2.2 Docker 安装（推荐）

```bash
# 拉取镜像
docker pull carlasim/carla:0.9.15

# 运行容器
docker run -it --gpus all --rm \
    -v /tmp/carla:/carla/L \
    -p 2000:2000 \
    carlasim/carla:0.9.15 \
    /bin/bash -c "./CarlaUE4.sh"
```

### 2.3 Windows 安装

1. 下载 CARLA 安装包（.exe）
2. 解压到指定目录
3. 运行 `CarlaUE4.exe`

## 三、Python API 基础

### 3.1 客户端连接

```python
import carla
import time

# 连接服务器
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# 获取世界
world = client.get_world()

# 获取蓝图库
blueprint_library = world.get_blueprint_library()

print(f"已连接到: {world.get_map().name}")
```

### 3.2 车辆控制

```python
# 获取车辆蓝图
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

# 设置生成点
spawn_points = world.get_map().get_spawn_points()
spawn_point = spawn_points[0]

# 生成车辆
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# 车辆控制
control = carla.VehicleControl(
    throttle=0.5,    # 油门 0-1
    steer=0.0,       # 转向 -1~1
    brake=0.0,       # 刹车 0-1
    hand_brake=False,
    reverse=False
)
vehicle.apply_control(control)

# 销毁车辆
vehicle.destroy()
```

### 3.3 传感器配置

```python
# === RGB 相机 ===
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1920')
camera_bp.set_attribute('image_size_y', '1080')
camera_bp.set_attribute('fov', '110')
camera_transform = carla.Transform(
    carla.Location(x=1.5, y=0.0, z=2.4),  # 位置
    carla.Rotation(pitch=-15)               # 俯仰角
)
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# 图像回调
def callback(image):
    # image 是一个 carla.Image 对象
    image.save_to_disk(f'output/{image.frame}.png')

camera.listen(callback)

# === LiDAR ===
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('range', '100')
lidar_bp.set_attribute('points_per_second', '100000')
lidar_transform = carla.Transform(carla.Location(z=2.5))
lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

def lidar_callback(data):
    # 获取点云数据
    points = data.get_array()
    print(f"点_point_cloud_as云数量: {len(points)}")

lidar.listen(lidar_callback)

# === GNSS ===
gnss_bp = blueprint_library.find('sensor.other.gnss')
gnss_transform = carla.Transform(carla.Location(z=1.0))
gnss = world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)

def gnss_callback(data):
    print(f"位置: lat={data.latitude}, lon={data.longitude}, alt={data.altitude}")

gnss.listen(gnss_callback)
```

## 四、场景构建

### 4.1 交通场景

```python
# 生成交通车辆
def spawn_traffic(world, blueprint_library, num_vehicles=50):
    vehicles = []
    vehicle_bps = blueprint_library.filter('vehicle.*')
    
    for i in range(num_vehicles):
        # 随机选择蓝图
        bp = random.choice(vehicle_bps)
        
        # 随机选择生成点
        spawn_point = random.choice(world.get_map().get_spawn_points())
        
        try:
            vehicle = world.spawn_actor(bp, spawn_point)
            vehicles.append(vehicle)
        except:
            continue
    
    return vehicles

# 设置自动巡航
for vehicle in traffic_vehicles:
    vehicle.set_autopilot(True)
```

### 4.2 行人场景

```python
# 生成行人
walker_bp = blueprint_library.filter('walker.pedestrian.*')
walker_controller_bp = blueprint_library.find('controller.ai.walker')

spawn_point = carla.Transform(
    carla.Location(x=-6, y=0, z=1.5),
    carla.Rotation(yaw=-90)
)

walker = world.spawn_actor(walker_bp[0], spawn_point)
controller = world.spawn_actor(walker_controller_bp, carla.Transform(), attach_to=walker)

# 行人控制
controller.start()
controller.go_to_location(carla.Location(x=10, y=0, z=1.5))
controller.set_max_speed(1.4)  # m/s
```

### 4.3 自定义天气

```python
# 设置天气
weather = world.get_weather()
weather.sun_altitude_angle = -30  # 太阳高度角
weather.cloudiness = 80           # 云量
weather.precipitation = 50        # 降水
weather.fog_density = 30           # 雾密度

world.set_weather(weather)
```

### 4.4 传感器同步

```python
# 设置固定时间步长（同步模式）
settings = world.get_settings()
settings.fixed_delta_seconds = 0.05  # 50ms
settings.synchronous_mode = True
world.apply_settings(settings)
```

## 五、场景编辑器

### 5.1 使用 OpenDRIVE 创建地图

```python
# 加载 OpenDRIVE 地图
world = client.generate_world(
    carla.MapLayer.NONE,  # 不使用默认层
    carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles
)

# 或者使用 Town
world = client.load_world('Town01')
```

### 5.2 自定义地图导出

```python
# 导出为 OpenDRIVE 格式
opendrive_xml = world.get_map().to_opendrive()
with open('my_map.xodr', 'w') as f:
    f.write(opendrive_xml)

# 加载自定义地图
client.generate_world('my_map.xodr')
```

## 六、实战案例

### 6.1 端到端自动驾驶测试

```python
import carla
import numpy as np
import pygame

# 初始化
client = carla.Client('localhost', 2000)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# 设置 pygame
pygame.init()
display = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()

# 创建车辆
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
vehicle = world.spawn_actor(
    vehicle_bp, 
    world.get_map().get_spawn_points()[0]
)

# 创建相机
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1280')
camera_bp.set_attribute('image_size_y', '720')
camera = world.spawn_actor(
    camera_bp,
    carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)),
    attach_to=vehicle
)

# 图像数组
image_array = []

def callback(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = array.reshape((image.height, image.width, 4))
    image_array.append(array[:, :, :3])

camera.listen(callback)

# 主循环
vehicle.set_autopilot(True)

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise StopIteration

        if image_array:
            img = image_array.pop(0)
            img = img[:, :, ::-1]  # BGR to RGB
            surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            display.blit(surface, (0, 0))
            pygame.display.flip()

        clock.tick(30)

finally:
    vehicle.destroy()
    camera.destroy()
    pygame.quit()
```

### 6.2 传感器数据采集

```python
import carla
import os

client = carla.Client('localhost', 2000)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# 创建输出目录
output_dir = '/data/carla_dataset'
os.makedirs(output_dir, exist_ok=True)

# 生成车辆
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
vehicle = world.spawn_actor(
    vehicle_bp,
    world.get_map().get_spawn_points()[0]
)

# 多个传感器
sensors = {}

# RGB相机
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1920')
camera_bp.set_attribute('image_size_y', '1080')
camera = world.spawn_actor(
    camera_bp,
    carla.Transform(carla.Location(x=1.5, z=2.4)),
    attach_to=vehicle
)
camera.listen(lambda img: img.save_to_disk(f'{output_dir}/camera/{img.frame}.png'))
sensors['camera'] = camera

# LiDAR
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('points_per_second', '100000')
lidar = world.spawn_actor(
    lidar_bp,
    carla.Transform(carla.Location(z=2.5)),
    attach_to=vehicle
)
lidar.listen(lambda data: data.save_to_disk(f'{output_dir}/lidar/{data.frame}.ply'))
sensors['lidar'] = lidar

# GPS
gnss_bp = blueprint_library.find('sensor.other.gnss')
gnss = world.spawn_actor(
    gnss_bp,
    carla.Transform(carla.Location(z=1.0)),
    attach_to=vehicle
)
gnss_listen = []

def gnss_callback(data, gnss_list=gnss_listen):
    gnss_list.append({
        'frame': data.frame,
        'lat': data.latitude,
        'lon': data.longitude,
        'alt': data.altitude
    })

gnss.listen(lambda data: gnss_callback(data))

# 采集数据
vehicle.set_autopilot(True)
import time
time.sleep(60)  # 采集60秒

# 清理
for sensor in sensors.values():
    sensor.destroy()
vehicle.destroy()

# 保存GPS数据
import json
with open(f'{output_dir}/gnss.json', 'w') as f:
    json.dump(gnss_listen, f, indent=2)
```

## 七、常见问题

### 7.1 启动失败

```bash
# 错误: libgtest.so not found
sudo apt-get install libgtest-dev

# 错误: Vulkan not supported
# 使用 OpenGL 启动
./CarlaUE4.sh -opengl
```

### 7.2 性能优化

```bash
# 降低画质提高帧率
./CarlaUE4.sh -quality-level=Low

# 无窗口模式
./CarlaUE4.sh -carla-server -RenderOffScreen
```

### 7.3 网络连接

```python
# 解决连接超时
client = carla.Client('localhost', 2000)
client.set_timeout(20.0)  # 增加超时时间

# 切换地图
world = client.load_world('Town02')
```

## 八、相关资源

- [CARLA 官方文档](https://carla.readthedocs.io/)
- [CARLA GitHub](https://github.com/carla-simulator/carla)
- [CARLA Python API](https://carla.readthedocs.io/en/latest/python_api/)
