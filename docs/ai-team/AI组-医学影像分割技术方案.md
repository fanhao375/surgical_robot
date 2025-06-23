# AI组技术方案 - 医学影像分割与路径规划

## 一、任务概述

### 1.1 目标

从DSA血管造影图像中自动提取血管中心线，生成导丝推送和旋转的轨迹序列。

### 1.2 输入输出定义

- **输入**：DSA图像（DICOM/PNG格式）
- **输出**：轨迹文件（CSV格式），包含推送距离和旋转角度序列

### 1.3 核心任务

1. 血管图像分割
2. 中心线提取
3. 路径规划
4. 轨迹生成

## 二、开发环境搭建（Day 1-2）

### 2.1 基础环境配置

```bash
# 1. 安装Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash Anaconda3-2023.09-0-Linux-x86_64.sh

# 2. 创建虚拟环境
conda create -n vessel_seg python=3.10
conda activate vessel_seg

# 3. 安装CUDA（如果有GPU）
# 检查CUDA版本
nvidia-smi

# 4. 安装PyTorch（根据CUDA版本选择）
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CPU版本
pip install torch torchvision torchaudio
```

### 2.2 医学影像处理库安装

```bash
# MONAI - 医学影像AI框架
pip install monai

# SimpleITK - 医学影像处理
pip install SimpleITK

# VMTK - 血管建模工具包
conda install -c vmtk vtk vmtk

# nnU-Net - 医学影像分割
pip install nnunet

# 其他依赖
pip install numpy pandas matplotlib scikit-image opencv-python
pip install nibabel pydicom scipy
```

### 2.3 开发工具配置

```bash
# Jupyter Lab
pip install jupyterlab

# 3D Slicer（用于数据标注）
# 下载地址：https://download.slicer.org/

# VS Code扩展
# - Python
# - Jupyter
# - DICOM Viewer
```

## 三、数据准备（Day 3-4）

### 3.1 数据收集方案

#### 选项1：公开数据集

```python
# 下载脚本
import requests
import zipfile

# 1. CTA脑血管数据集（可作为初步测试）
url = "https://github.com/datasets/vessel/cta_brain.zip"

# 2. 冠脉造影数据集
# 申请地址：https://www.physionet.org/

# 3. 合成数据生成
from vessel_synthesis import generate_synthetic_vessels
synthetic_data = generate_synthetic_vessels(
    num_samples=50,
    vessel_types=['coronary', 'cerebral']
)
```

#### 选项2：硅胶模型拍摄

```python
# 数据采集协议
"""
1. 拍摄角度：LAO 30°, RAO 30°, AP
2. 造影剂浓度：1:1稀释
3. 帧率：15 fps
4. 分辨率：512x512或1024x1024
5. 保存格式：DICOM
"""
```

### 3.2 数据标注流程

#### 使用3D Slicer标注

```python
# 标注步骤文档
"""
1. 加载DICOM数据
   File -> Add Data -> Choose DICOM
   
2. 血管分割
   - Segment Editor模块
   - Threshold工具：设置血管灰度范围
   - Islands工具：保留最大连通域
   - Smoothing：平滑血管边缘
   
3. 导出标注
   - 保存为.nii.gz格式
   - 命名规则：case_001_vessel.nii.gz
"""

# 批量转换脚本
import SimpleITK as sitk
import os

def convert_dicom_to_nifti(dicom_dir, output_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, output_path)
```

### 3.3 数据组织结构

```
vessel_dataset/
├── raw_data/
│   ├── case_001/
│   │   ├── image.nii.gz
│   │   └── mask.nii.gz
│   ├── case_002/
│   └── ...
├── preprocessed/
│   ├── images/
│   └── masks/
└── splits/
    ├── train.txt  # 70%
    ├── val.txt    # 15%
    └── test.txt   # 15%
```

## 四、血管分割模型（Day 5-8）

### 4.1 使用nnU-Net训练

#### 数据预处理

```bash
# 1. 转换数据格式
nnUNet_convert_decathlon_task -i vessel_dataset

# 2. 规划预处理
nnUNet_plan_and_preprocess -t 501 --verify_dataset_integrity

# 3. 查看数据集信息
nnUNet_print_available_pretrained_models
```

#### 模型训练

```bash
# 训练2D模型（快速验证）
nnUNet_train 2d nnUNetTrainerV2 Task501_Vessel 0 --npz

# 训练3D模型（精度更高）
nnUNet_train 3d_fullres nnUNetTrainerV2 Task501_Vessel 0 --npz

# 低显存版本
nnUNet_train 3d_lowres nnUNetTrainerV2 Task501_Vessel 0 --npz
```

#### 训练监控脚本

```python
# monitor_training.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_training_progress(log_file):
    # 读取训练日志
    data = pd.read_csv(log_file, sep='\t')
  
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
  
    # Loss曲线
    axes[0,0].plot(data['epoch'], data['train_loss'], label='Train')
    axes[0,0].plot(data['epoch'], data['val_loss'], label='Val')
    axes[0,0].set_title('Loss')
    axes[0,0].legend()
  
    # Dice系数
    axes[0,1].plot(data['epoch'], data['dice_score'])
    axes[0,1].set_title('Dice Score')
  
    # 学习率
    axes[1,0].plot(data['epoch'], data['learning_rate'])
    axes[1,0].set_title('Learning Rate')
  
    plt.tight_layout()
    plt.savefig('training_progress.png')
```

### 4.2 自定义分割模型（备选方案）

```python
# vessel_segmentation.py
import torch
import torch.nn as nn
from monai.networks.nets import UNet

class VesselSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
      
    def forward(self, x):
        return self.model(x)

# 训练脚本
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from torch.optim import Adam

def train_model(model, train_loader, val_loader, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
  
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = Adam(model.parameters(), lr=1e-4)
    metric = DiceMetric(include_background=False, reduction="mean")
  
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
      
        for batch_data in train_loader:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
          
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
          
            epoch_loss += loss.item()
      
        # 验证
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                val_outputs = model(val_inputs)
                metric(y_pred=val_outputs, y=val_labels)
          
            dice_score = metric.aggregate().item()
            metric.reset()
      
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Dice: {dice_score:.4f}")
```

## 五、中心线提取（Day 9-10）

### 5.1 使用VMTK提取中心线

```python
# centerline_extraction.py
import vmtk
import SimpleITK as sitk
import numpy as np

def extract_centerline(mask_path, output_path):
    """
    从血管分割结果提取中心线
    """
    # 读取分割结果
    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask)
  
    # VMTK中心线提取
    from vmtk import vmtkscripts
  
    # 1. 转换为表面网格
    surface = vmtkscripts.vmtkMarchingCubes()
    surface.Image = mask
    surface.Level = 0.5
    surface.Execute()
  
    # 2. 提取中心线
    centerlines = vmtkscripts.vmtkCenterlines()
    centerlines.Surface = surface.Surface
    centerlines.SeedSelectorName = 'pointlist'
    centerlines.SourcePoints = [0, 0, 0]  # 起点
    centerlines.TargetPoints = [100, 100, 100]  # 终点
    centerlines.Execute()
  
    # 3. 保存中心线点
    points = centerlines.Centerlines.GetPoints()
    centerline_points = []
    for i in range(points.GetNumberOfPoints()):
        point = points.GetPoint(i)
        centerline_points.append(point)
  
    # 保存为CSV
    import pandas as pd
    df = pd.DataFrame(centerline_points, columns=['x', 'y', 'z'])
    df.to_csv(output_path, index=False)
  
    return centerline_points
```

### 5.2 简化版中心线提取（骨架化）

```python
# skeleton_extraction.py
from skimage.morphology import skeletonize, thin
import numpy as np
from scipy import ndimage

def extract_skeleton(binary_mask):
    """
    使用骨架化算法提取中心线
    """
    # 细化处理
    skeleton = skeletonize(binary_mask)
  
    # 提取骨架点
    points = np.column_stack(np.where(skeleton))
  
    # 排序点形成路径
    ordered_points = order_skeleton_points(points)
  
    return ordered_points

def order_skeleton_points(points):
    """
    将骨架点排序成连续路径
    """
    from sklearn.neighbors import NearestNeighbors
  
    if len(points) < 2:
        return points
  
    # 找到端点（度为1的点）
    nbrs = NearestNeighbors(n_neighbors=3, radius=1.5).fit(points)
    distances, indices = nbrs.kneighbors(points)
  
    # 计算每个点的度
    degrees = np.sum(distances < 1.5, axis=1) - 1
    endpoints = points[degrees == 1]
  
    if len(endpoints) == 0:
        endpoints = [points[0]]
  
    # 从第一个端点开始排序
    ordered = [endpoints[0]]
    remaining = set(range(len(points)))
    remaining.remove(np.where((points == endpoints[0]).all(axis=1))[0][0])
  
    while remaining:
        last_point = ordered[-1]
        distances = np.linalg.norm(points[list(remaining)] - last_point, axis=1)
        nearest_idx = list(remaining)[np.argmin(distances)]
        ordered.append(points[nearest_idx])
        remaining.remove(nearest_idx)
  
    return np.array(ordered)
```

## 六、路径规划（Day 11-12）

### 6.1 基于规则的路径规划

```python
# path_planning.py
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

class PathPlanner:
    def __init__(self, centerline_points, sampling_distance=0.5):
        """
        centerline_points: 中心线点坐标 (N, 3)
        sampling_distance: 采样间隔 (mm)
        """
        self.centerline = np.array(centerline_points)
        self.sampling_distance = sampling_distance
      
    def generate_trajectory(self):
        """
        生成导丝轨迹
        """
        # 1. 平滑中心线
        smoothed_path = self._smooth_path()
      
        # 2. 重采样
        resampled_path = self._resample_path(smoothed_path)
      
        # 3. 计算推送和旋转
        trajectory = self._compute_trajectory(resampled_path)
      
        return trajectory
  
    def _smooth_path(self):
        """
        使用样条插值平滑路径
        """
        # 参数化
        t = np.linspace(0, 1, len(self.centerline))
      
        # 对每个维度进行样条插值
        cs_x = CubicSpline(t, self.centerline[:, 0])
        cs_y = CubicSpline(t, self.centerline[:, 1])
        cs_z = CubicSpline(t, self.centerline[:, 2])
      
        # 生成平滑路径
        t_smooth = np.linspace(0, 1, len(self.centerline) * 5)
        smooth_path = np.column_stack([
            cs_x(t_smooth),
            cs_y(t_smooth),
            cs_z(t_smooth)
        ])
      
        return smooth_path
  
    def _resample_path(self, path):
        """
        按固定间隔重采样路径
        """
        # 计算累积距离
        distances = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        distances = np.insert(distances, 0, 0)
      
        # 按固定间隔采样
        total_length = distances[-1]
        num_samples = int(total_length / self.sampling_distance)
        sample_distances = np.linspace(0, total_length, num_samples)
      
        # 插值得到采样点
        resampled = np.zeros((num_samples, 3))
        for i in range(3):
            resampled[:, i] = np.interp(sample_distances, distances, path[:, i])
      
        return resampled
  
    def _compute_trajectory(self, path):
        """
        计算推送距离和旋转角度
        """
        trajectory = []
      
        for i in range(1, len(path)):
            # 推送距离
            push_distance = np.linalg.norm(path[i] - path[i-1])
          
            # 计算切向量
            if i < len(path) - 1:
                tangent = path[i+1] - path[i-1]
            else:
                tangent = path[i] - path[i-1]
          
            tangent = tangent / np.linalg.norm(tangent)
          
            # 计算旋转角度（相对于初始方向）
            if i == 1:
                self.initial_direction = tangent
                rotation_angle = 0
            else:
                # 计算旋转轴和角度
                axis = np.cross(self.initial_direction, tangent)
                if np.linalg.norm(axis) > 1e-6:
                    axis = axis / np.linalg.norm(axis)
                    angle = np.arccos(np.clip(np.dot(self.initial_direction, tangent), -1, 1))
                    rotation_angle = np.degrees(angle)
                else:
                    rotation_angle = 0
          
            # 速度规划（简单梯形速度）
            push_velocity = min(5.0, push_distance * 10)  # mm/s
            angular_velocity = min(30.0, abs(rotation_angle) * 2)  # deg/s
          
            trajectory.append({
                'push_mm': push_distance,
                'rotate_deg': rotation_angle,
                'velocity_mm_s': push_velocity,
                'angular_velocity_deg_s': angular_velocity,
                'position': path[i].tolist()
            })
      
        return trajectory
```

### 6.2 生成控制组所需的轨迹文件

```python
# trajectory_generator.py
import csv
import numpy as np

class TrajectoryGenerator:
    def __init__(self):
        self.time_step = 100  # ms
      
    def save_trajectory(self, trajectory_points, output_file):
        """
        将轨迹保存为控制组所需的CSV格式
        """
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # 写入标题行
            writer.writerow(['time_ms', 'push_mm', 'rotate_deg', 
                           'velocity_mm_s', 'angular_velocity_deg_s'])
          
            time_ms = 0
            cumulative_push = 0
            cumulative_rotation = 0
          
            for point in trajectory_points:
                cumulative_push += point['push_mm']
                cumulative_rotation += point['rotate_deg']
              
                writer.writerow([
                    time_ms,
                    round(cumulative_push, 3),
                    round(cumulative_rotation, 3),
                    round(point['velocity_mm_s'], 3),
                    round(point['angular_velocity_deg_s'], 3)
                ])
              
                # 根据速度计算下一个时间点
                time_increment = max(
                    point['push_mm'] / point['velocity_mm_s'] * 1000,
                    abs(point['rotate_deg']) / point['angular_velocity_deg_s'] * 1000
                )
                time_ms += int(time_increment)
      
        print(f"轨迹已保存到: {output_file}")
        print(f"总时长: {time_ms/1000:.1f} 秒")
        print(f"总推送: {cumulative_push:.1f} mm")
        print(f"总旋转: {cumulative_rotation:.1f} 度")
```

## 七、推理服务（Day 13）

### 7.1 模型推理API

```python
# inference_server.py
from flask import Flask, request, jsonify
import torch
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import tempfile
import time

app = Flask(__name__)

class InferenceEngine:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.path_planner = PathPlanner(sampling_distance=0.5)
      
    def load_model(self, model_path):
        # 加载nnUNet模型
        from nnunet.inference.predict import predict_from_folder
        return model_path
      
    def segment_vessel(self, image_path):
        """
        血管分割
        """
        start_time = time.time()
      
        # 使用nnUNet推理
        output_folder = tempfile.mkdtemp()
        predict_from_folder(
            model_folder=self.model,
            input_folder=str(Path(image_path).parent),
            output_folder=output_folder,
            save_npz=False,
            num_threads_preprocessing=2,
            num_threads_nifti_save=2,
            fold='all',
            mode='normal'
        )
      
        # 读取分割结果
        mask_path = list(Path(output_folder).glob("*.nii.gz"))[0]
        mask = sitk.ReadImage(str(mask_path))
      
        inference_time = time.time() - start_time
        print(f"分割耗时: {inference_time:.2f}秒")
      
        return mask
      
    def extract_trajectory(self, mask):
        """
        从分割结果提取轨迹
        """
        # 提取中心线
        centerline = extract_centerline(mask)
      
        # 生成轨迹
        trajectory = self.path_planner.generate_trajectory(centerline)
      
        return trajectory

# 创建推理引擎实例
engine = InferenceEngine("/path/to/nnunet_model")

@app.route('/segment', methods=['POST'])
def segment_and_plan():
    """
    API端点：输入图像，返回轨迹
    """
    try:
        # 获取上传的图像
        image_file = request.files['image']
      
        # 保存临时文件
        temp_path = tempfile.mktemp(suffix='.nii.gz')
        image_file.save(temp_path)
      
        # 分割血管
        mask = engine.segment_vessel(temp_path)
      
        # 提取轨迹
        trajectory = engine.extract_trajectory(mask)
      
        # 生成CSV文件
        output_file = tempfile.mktemp(suffix='.csv')
        generator = TrajectoryGenerator()
        generator.save_trajectory(trajectory, output_file)
      
        # 读取CSV内容
        with open(output_file, 'r') as f:
            csv_content = f.read()
      
        return jsonify({
            'status': 'success',
            'trajectory_csv': csv_content,
            'num_points': len(trajectory),
            'total_push_mm': sum(p['push_mm'] for p in trajectory),
            'total_rotation_deg': sum(p['rotate_deg'] for p in trajectory)
        })
      
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

### 7.2 批处理脚本

```python
# batch_process.py
import os
from pathlib import Path
import pandas as pd

def batch_process_images(input_dir, output_dir, model_path):
    """
    批量处理图像生成轨迹
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
  
    # 初始化推理引擎
    engine = InferenceEngine(model_path)
  
    # 处理结果记录
    results = []
  
    for image_file in input_path.glob("*.nii.gz"):
        print(f"处理: {image_file.name}")
      
        try:
            # 分割
            mask = engine.segment_vessel(str(image_file))
          
            # 生成轨迹
            trajectory = engine.extract_trajectory(mask)
          
            # 保存轨迹
            output_file = output_path / f"{image_file.stem}_trajectory.csv"
            generator = TrajectoryGenerator()
            generator.save_trajectory(trajectory, str(output_file))
          
            # 记录结果
            results.append({
                'image': image_file.name,
                'status': 'success',
                'num_points': len(trajectory),
                'total_push_mm': sum(p['push_mm'] for p in trajectory),
                'total_rotation_deg': sum(p['rotate_deg'] for p in trajectory)
            })
          
        except Exception as e:
            results.append({
                'image': image_file.name,
                'status': 'failed',
                'error': str(e)
            })
  
    # 保存处理报告
    df = pd.DataFrame(results)
    df.to_csv(output_path / 'processing_report.csv', index=False)
  
    print(f"\n处理完成！")
    print(f"成功: {len(df[df['status']=='success'])}")
    print(f"失败: {len(df[df['status']=='failed'])}")
```

## 八、性能优化（Day 14）

### 8.1 模型加速

```python
# model_optimization.py
import torch
import onnx
import tensorrt as trt

def optimize_model_tensorrt(pytorch_model, dummy_input, output_path):
    """
    使用TensorRT优化模型
    """
    # 1. 导出ONNX
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        "temp_model.onnx",
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}}
    )
  
    # 2. 构建TensorRT引擎
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, logger)
  
    with open("temp_model.onnx", 'rb') as model:
        parser.parse(model.read())
  
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    config.set_flag(trt.BuilderFlag.FP16)  # 使用FP16
  
    engine = builder.build_engine(network, config)
  
    # 保存引擎
    with open(output_path, 'wb') as f:
        f.write(engine.serialize())
  
    return engine

def benchmark_inference_speed(model, test_data):
    """
    测试推理速度
    """
    import time
  
    # 预热
    for _ in range(10):
        _ = model(test_data)
  
    # 计时
    times = []
    for _ in range(100):
        start = time.time()
        _ = model(test_data)
        times.append(time.time() - start)
  
    print(f"平均推理时间: {np.mean(times)*1000:.2f} ms")
    print(f"标准差: {np.std(times)*1000:.2f} ms")
    print(f"最小/最大: {np.min(times)*1000:.2f}/{np.max(times)*1000:.2f} ms")
```

### 8.2 并行处理优化

```python
# parallel_processing.py
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

class ParallelProcessor:
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or mp.cpu_count()
      
    def process_batch(self, images, model):
        """
        并行处理多张图像
        """
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for image in images:
                future = executor.submit(self.process_single, image, model)
                futures.append(future)
          
            results = []
            for future in futures:
                result = future.result()
                results.append(result)
      
        return results
  
    def process_single(self, image, model):
        """
        处理单张图像
        """
        # 分割
        mask = model.segment(image)
      
        # 提取中心线
        centerline = extract_centerline(mask)
      
        # 生成轨迹
        trajectory = generate_trajectory(centerline)
      
        return trajectory
```

## 九、测试与验证（Day 14）

### 9.1 单元测试

```python
# test_pipeline.py
import unittest
import numpy as np

class TestVesselSegmentation(unittest.TestCase):
    def setUp(self):
        self.test_image = np.random.randn(1, 1, 128, 128, 64)
        self.model = load_test_model()
      
    def test_segmentation_output_shape(self):
        """测试分割输出形状"""
        output = self.model(self.test_image)
        self.assertEqual(output.shape, self.test_image.shape)
      
    def test_centerline_extraction(self):
        """测试中心线提取"""
        mask = np.zeros((100, 100, 100))
        # 创建简单的管状结构
        mask[40:60, 40:60, :] = 1
      
        centerline = extract_centerline(mask)
        self.assertGreater(len(centerline), 10)
      
    def test_trajectory_generation(self):
        """测试轨迹生成"""
        # 创建测试中心线
        centerline = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 2],
            [1, 2, 3]
        ])
      
        planner = PathPlanner(centerline)
        trajectory = planner.generate_trajectory()
      
        # 验证轨迹点数量
        self.assertGreater(len(trajectory), 0)
      
        # 验证推送距离为正
        for point in trajectory:
            self.assertGreaterEqual(point['push_mm'], 0)

if __name__ == '__main__':
    unittest.main()
```

### 9.2 集成测试数据

```python
# generate_test_data.py
def generate_test_cases():
    """
    生成测试用例
    """
    test_cases = [
        {
            'name': 'straight_vessel',
            'description': '直血管',
            'centerline': generate_straight_line(length=50),
            'expected_rotation': 0
        },
        {
            'name': 'curved_vessel', 
            'description': '弯曲血管',
            'centerline': generate_curved_line(radius=20, angle=90),
            'expected_rotation': 90
        },
        {
            'name': 'spiral_vessel',
            'description': '螺旋血管', 
            'centerline': generate_spiral(radius=10, pitch=5, turns=2),
            'expected_rotation': 720
        }
    ]
  
    return test_cases

def generate_straight_line(length):
    """生成直线"""
    return np.array([[0, 0, i] for i in range(length)])

def generate_curved_line(radius, angle):
    """生成弧线"""
    theta = np.linspace(0, np.radians(angle), 50)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.linspace(0, 10, 50)
    return np.column_stack([x, y, z])
```

## 十、与控制组对接

### 10.1 接口文档

```markdown
# AI组输出接口文档

## 轨迹文件格式
- 文件类型：CSV
- 编码：UTF-8
- 分隔符：逗号

## 字段说明
| 字段名 | 类型 | 单位 | 说明 |
|--------|------|------|------|
| time_ms | int | 毫秒 | 时间戳 |
| push_mm | float | 毫米 | 绝对推送位置 |
| rotate_deg | float | 度 | 绝对旋转角度 |
| velocity_mm_s | float | 毫米/秒 | 推送速度 |
| angular_velocity_deg_s | float | 度/秒 | 旋转速度 |

## 约束条件
- 最大推送速度：10 mm/s
- 最大旋转速度：30 deg/s
- 时间间隔：建议100ms
- 位置精度：0.1mm
- 角度精度：0.1度
```

### 10.2 验证工具

```python
# validate_trajectory.py
def validate_trajectory_file(csv_file):
    """
    验证轨迹文件是否符合规范
    """
    import pandas as pd
  
    # 读取CSV
    df = pd.read_csv(csv_file)
  
    # 检查必需字段
    required_fields = ['time_ms', 'push_mm', 'rotate_deg', 
                      'velocity_mm_s', 'angular_velocity_deg_s']
    missing_fields = set(required_fields) - set(df.columns)
    if missing_fields:
        return False, f"缺少字段: {missing_fields}"
  
    # 检查数据类型
    if not pd.api.types.is_numeric_dtype(df['time_ms']):
        return False, "time_ms必须是数值类型"
  
    # 检查时间单调性
    if not df['time_ms'].is_monotonic_increasing:
        return False, "时间戳必须单调递增"
  
    # 检查速度限制
    if (df['velocity_mm_s'] > 10).any():
        return False, "推送速度超过限制(10mm/s)"
  
    if (df['angular_velocity_deg_s'] > 30).any():
        return False, "旋转速度超过限制(30deg/s)"
  
    # 检查位置连续性
    push_diff = df['push_mm'].diff().abs()
    if (push_diff > 5).any():  # 5mm跳变检查
        return False, "推送位置存在异常跳变"
  
    return True, "验证通过"

# 使用示例
if __name__ == "__main__":
    result, message = validate_trajectory_file("test_trajectory.csv")
    print(f"验证结果: {message}")
```

## 十一、常见问题解决

### Q1: GPU内存不足

```python
# 解决方案1：减小batch size
train_config['batch_size'] = 1

# 解决方案2：使用混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

# 解决方案3：使用gradient checkpointing
model.gradient_checkpointing_enable()
```

### Q2: 分割效果不佳

```python
# 数据增强
from monai.transforms import (
    Compose, LoadImaged, AddChanneld,
    RandRotate90d, RandFlipd, RandZoomd,
    NormalizeIntensityd, RandGaussianNoised
)

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),
    RandRotate90d(keys=["image", "label"], prob=0.5),
    RandFlipd(keys=["image", "label"], prob=0.5),
    RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1),
    RandGaussianNoised(keys=["image"], prob=0.5),
    NormalizeIntensityd(keys=["image"])
])
```

### Q3: 中心线提取失败

```python
# 备选方案：使用距离变换
from scipy.ndimage import distance_transform_edt

def extract_centerline_dt(binary_mask):
    # 距离变换
    dist = distance_transform_edt(binary_mask)
  
    # 找到局部最大值（脊线）
    from skimage.feature import peak_local_max
    peaks = peak_local_max(dist, min_distance=5)
  
    return peaks
```

## 十二、交付清单

### 12.1 模型交付

- [ ] 训练好的分割模型（.pth文件）
- [ ] 模型配置文件（yaml）
- [ ] 模型推理脚本

### 12.2 代码交付

- [ ] 数据预处理脚本
- [ ] 训练脚本
- [ ] 推理服务API
- [ ] 轨迹生成工具

### 12.3 文档交付

- [ ] 数据标注指南
- [ ] 模型训练手册
- [ ] API接口文档
- [ ] 故障排查指南

### 12.4 测试数据

- [ ] 示例输入图像（5-10个）
- [ ] 对应的轨迹输出
- [ ] 性能测试报告

---

**下一步行动（AI组）**：

1. **今天**：搭建Python环境，安装MONAI和nnUNet
2. **明天**：准备10个血管图像样本
3. **第3天**：完成第一次分割模型训练
4. **第1周末**：实现中心线提取
5. **第2周**：生成第一批轨迹文件供控制组测试