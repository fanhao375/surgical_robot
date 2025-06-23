# AI组 - 第一周任务清单

## 任务概览

**目标**：完成AI开发环境搭建，准备训练数据，初步验证血管分割流程

**关键成果**：

1. Python深度学习环境配置完成
2. 准备10-20个标注样本
3. nnUNet训练流程跑通
4. 生成第一个测试轨迹文件

---

## Day 1（周一）：Python环境搭建

### 上午：Anaconda和CUDA环境

```bash
# 1. 下载并安装Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash Anaconda3-2023.09-0-Linux-x86_64.sh

# 2. 配置conda
conda config --add channels defaults
conda config --add channels conda-forge
conda config --set channel_priority strict

# 3. 创建项目环境
conda create -n vessel_seg python=3.10
conda activate vessel_seg

# 4. 检查GPU（如果有）
nvidia-smi
```

### 下午：深度学习框架安装

```bash
# 1. 安装PyTorch（根据CUDA版本选择）
# CUDA 11.8版本
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# CPU版本（如果没有GPU）
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# 2. 验证安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"

# 3. 安装基础包
pip install numpy pandas matplotlib scikit-learn opencv-python
pip install jupyter notebook ipywidgets
```

### 验收标准

- [ ] `conda env list` 显示vessel_seg环境
- [ ] PyTorch导入无错误
- [ ] GPU可用（如果有显卡）

---

## Day 2（周二）：医学影像处理环境

### 上午：MONAI和相关工具安装

```bash
# 1. 安装MONAI
pip install monai

# 2. 安装SimpleITK
pip install SimpleITK

# 3. 安装nibabel（NIfTI格式支持）
pip install nibabel pydicom

# 4. 安装可视化工具
pip install plotly kaleido

# 5. 验证MONAI安装
python -c "import monai; print(f'MONAI版本: {monai.__version__}')"
```

### 下午：nnU-Net安装配置

```bash
# 1. 安装nnU-Net
pip install nnunet

# 2. 设置环境变量（添加到~/.bashrc）
export nnUNet_raw_data_base="/home/$USER/nnUNet_raw_data_base"
export nnUNet_preprocessed="/home/$USER/nnUNet_preprocessed"
export RESULTS_FOLDER="/home/$USER/nnUNet_trained_models"

# 3. 创建必要目录
mkdir -p $nnUNet_raw_data_base
mkdir -p $nnUNet_preprocessed
mkdir -p $RESULTS_FOLDER

# 4. 验证安装
nnUNet_print_available_pretrained_models
```

创建测试脚本 `test_environment.py`：

```python
#!/usr/bin/env python3
"""环境测试脚本"""

import sys
print("Python版本:", sys.version)

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("✗ PyTorch未安装")

try:
    import monai
    print(f"✓ MONAI {monai.__version__}")
except ImportError:
    print("✗ MONAI未安装")

try:
    import SimpleITK as sitk
    print(f"✓ SimpleITK {sitk.Version_VersionString()}")
except ImportError:
    print("✗ SimpleITK未安装")

try:
    import nibabel as nib
    print(f"✓ NiBabel {nib.__version__}")
except ImportError:
    print("✗ NiBabel未安装")

print("\n环境检查完成！")
```

### 验收标准

- [ ] 所有库导入无错误
- [ ] nnU-Net命令可用
- [ ] 环境变量设置正确

---

## Day 3（周三）：数据准备和标注

### 上午：准备示例数据

```python
# generate_synthetic_vessels.py
"""生成合成血管数据用于测试"""

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
import os

def create_3d_vessel(size=(128, 128, 64), vessel_radius=3):
    """创建3D血管模拟数据"""
    volume = np.zeros(size)
  
    # 创建简单的管状结构
    z_center = size[2] // 2
    y_center = size[1] // 2
    x_start = size[0] // 4
    x_end = 3 * size[0] // 4
  
    # 主血管
    for x in range(x_start, x_end):
        # 添加一些弯曲
        y_offset = int(5 * np.sin(x * 0.1))
        y = y_center + y_offset
      
        # 创建圆形截面
        for dy in range(-vessel_radius, vessel_radius + 1):
            for dz in range(-vessel_radius, vessel_radius + 1):
                if dy**2 + dz**2 <= vessel_radius**2:
                    if 0 <= y + dy < size[1] and 0 <= z_center + dz < size[2]:
                        volume[x, y + dy, z_center + dz] = 1
  
    # 添加分支
    branch_start = size[0] // 2
    for x in range(branch_start, x_end):
        y = y_center + int(5 * np.sin(branch_start * 0.1)) + (x - branch_start) // 2
        if y < size[1]:
            for dy in range(-2, 3):
                for dz in range(-2, 3):
                    if dy**2 + dz**2 <= 4:
                        if 0 <= y + dy < size[1] and 0 <= z_center + dz < size[2]:
                            volume[x, y + dy, z_center + dz] = 1
  
    # 平滑处理
    volume = gaussian_filter(volume, sigma=1)
  
    # 添加噪声和背景
    noise = np.random.normal(0, 0.1, size)
    image = volume * 200 + 50 + noise * 10
    image = np.clip(image, 0, 255)
  
    return image.astype(np.float32), volume.astype(np.uint8)

def save_sample(image, mask, sample_id, output_dir):
    """保存为NIfTI格式"""
    os.makedirs(output_dir, exist_ok=True)
  
    # 保存图像
    image_sitk = sitk.GetImageFromArray(image)
    image_sitk.SetSpacing([1.0, 1.0, 1.0])
    sitk.WriteImage(image_sitk, os.path.join(output_dir, f"vessel_{sample_id:03d}.nii.gz"))
  
    # 保存标注
    mask_sitk = sitk.GetImageFromArray(mask)
    mask_sitk.SetSpacing([1.0, 1.0, 1.0])
    sitk.WriteImage(mask_sitk, os.path.join(output_dir, f"vessel_{sample_id:03d}_seg.nii.gz"))

# 生成10个样本
output_base = os.path.expanduser("~/vessel_dataset")
for i in range(10):
    print(f"生成样本 {i+1}/10")
    image, mask = create_3d_vessel()
    save_sample(image, mask, i+1, output_base)

print(f"数据已保存到: {output_base}")
```

### 下午：数据可视化和验证

创建 `visualize_data.py`：

```python
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import os

def visualize_vessel(image_path, mask_path=None):
    """可视化血管图像和标注"""
    # 读取数据
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
  
    if mask_path:
        mask = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask)
  
    # 选择中间切片
    slice_idx = image_array.shape[0] // 2
  
    # 绘图
    if mask_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
      
        # 原始图像
        axes[0].imshow(image_array[slice_idx], cmap='gray')
        axes[0].set_title('原始图像')
        axes[0].axis('off')
      
        # 标注
        axes[1].imshow(mask_array[slice_idx], cmap='hot')
        axes[1].set_title('血管标注')
        axes[1].axis('off')
      
        # 叠加
        axes[2].imshow(image_array[slice_idx], cmap='gray')
        axes[2].imshow(mask_array[slice_idx], cmap='hot', alpha=0.5)
        axes[2].set_title('叠加显示')
        axes[2].axis('off')
    else:
        plt.imshow(image_array[slice_idx], cmap='gray')
        plt.title('血管图像')
        plt.axis('off')
  
    plt.tight_layout()
    plt.show()

# 可视化第一个样本
data_dir = os.path.expanduser("~/vessel_dataset")
visualize_vessel(
    os.path.join(data_dir, "vessel_001.nii.gz"),
    os.path.join(data_dir, "vessel_001_seg.nii.gz")
)
```

### 验收标准

- [ ] 生成10个合成血管样本
- [ ] 能够正确可视化
- [ ] 数据格式符合nnU-Net要求

---

## Day 4（周四）：nnU-Net数据准备

### 上午：组织数据为nnU-Net格式

```bash
# 创建任务文件夹
cd $nnUNet_raw_data_base
mkdir -p Task501_VesselSeg
cd Task501_VesselSeg
mkdir -p imagesTr labelsTr imagesTs

# 数据组织脚本
```

创建 `prepare_nnunet_data.py`：

```python
import os
import shutil
import json
from pathlib import Path

def prepare_nnunet_dataset():
    # 路径设置
    source_dir = Path.home() / "vessel_dataset"
    nnunet_base = Path(os.environ['nnUNet_raw_data_base'])
    task_dir = nnunet_base / "Task501_VesselSeg"
  
    # 创建目录
    (task_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (task_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    (task_dir / "imagesTs").mkdir(parents=True, exist_ok=True)
  
    # 复制文件（8个训练，2个测试）
    train_cases = list(range(1, 9))
    test_cases = list(range(9, 11))
  
    # 训练数据
    for case_id in train_cases:
        # 图像
        src_img = source_dir / f"vessel_{case_id:03d}.nii.gz"
        dst_img = task_dir / "imagesTr" / f"vessel_{case_id:03d}_0000.nii.gz"
        shutil.copy2(src_img, dst_img)
      
        # 标注
        src_lbl = source_dir / f"vessel_{case_id:03d}_seg.nii.gz"
        dst_lbl = task_dir / "labelsTr" / f"vessel_{case_id:03d}.nii.gz"
        shutil.copy2(src_lbl, dst_lbl)
  
    # 测试数据
    for case_id in test_cases:
        src_img = source_dir / f"vessel_{case_id:03d}.nii.gz"
        dst_img = task_dir / "imagesTs" / f"vessel_{case_id:03d}_0000.nii.gz"
        shutil.copy2(src_img, dst_img)
  
    # 创建dataset.json
    dataset_info = {
        "name": "VesselSegmentation",
        "description": "血管分割任务",
        "tensorImageSize": "4D",
        "reference": "internal",
        "licence": "internal",
        "release": "0.0",
        "modality": {
            "0": "CT"
        },
        "labels": {
            "0": "background",
            "1": "vessel"
        },
        "numTraining": len(train_cases),
        "numTest": len(test_cases),
        "training": [
            {
                "image": f"./imagesTr/vessel_{case_id:03d}_0000.nii.gz",
                "label": f"./labelsTr/vessel_{case_id:03d}.nii.gz"
            }
            for case_id in train_cases
        ],
        "test": [f"./imagesTs/vessel_{case_id:03d}_0000.nii.gz" for case_id in test_cases]
    }
  
    with open(task_dir / "dataset.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)
  
    print(f"数据集准备完成: {task_dir}")
    print(f"训练样本: {len(train_cases)}")
    print(f"测试样本: {len(test_cases)}")

if __name__ == "__main__":
    prepare_nnunet_dataset()
```

### 下午：nnU-Net预处理

```bash
# 1. 验证数据集
nnUNet_plan_and_preprocess -t 501 --verify_dataset_integrity

# 2. 如果验证通过，会自动进行预处理
# 这个过程可能需要一些时间

# 3. 检查预处理结果
ls $nnUNet_preprocessed/Task501_VesselSeg
```

### 验收标准

- [ ] dataset.json创建正确
- [ ] 数据验证通过
- [ ] 预处理完成

---

## Day 5（周五）：模型训练和轨迹生成

### 上午：启动nnU-Net训练

```bash
# 1. 开始训练（2D快速测试）
nnUNet_train 2d nnUNetTrainerV2 Task501_VesselSeg 0 --npz

# 注意：完整训练需要较长时间，可以先训练20个epoch测试
nnUNet_train 2d nnUNetTrainerV2 Task501_VesselSeg 0 --npz -e 20

# 2. 监控训练进度
# 新终端查看日志
tail -f $RESULTS_FOLDER/nnUNet/2d/Task501_VesselSeg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/training_log_*.txt
```

创建训练监控脚本 `monitor_training.py`：

```python
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path

def plot_training_curves():
    """实时绘制训练曲线"""
    results_folder = Path(os.environ['RESULTS_FOLDER'])
    log_file = list(results_folder.glob("**/training_log_*.txt"))
  
    if not log_file:
        print("未找到训练日志")
        return
  
    log_file = log_file[0]
  
    # 读取日志
    epochs = []
    train_losses = []
    val_losses = []
  
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'epoch:' in line:
                parts = line.split()
                epoch = int(parts[1])
                train_loss = float(parts[3])
                epochs.append(epoch)
                train_losses.append(train_loss)
  
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('nnU-Net训练进度')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.show()

if __name__ == "__main__":
    plot_training_curves()
```

### 下午：简单轨迹生成测试

创建 `generate_test_trajectory.py`：

```python
"""生成测试轨迹文件供控制组使用"""

import numpy as np
import pandas as pd
from pathlib import Path

def extract_centerline_simple(mask_array):
    """简化的中心线提取（用于测试）"""
    # 找到血管点
    vessel_points = np.argwhere(mask_array > 0)
  
    if len(vessel_points) == 0:
        return []
  
    # 按x坐标排序
    vessel_points = vessel_points[vessel_points[:, 0].argsort()]
  
    # 简单降采样
    step = max(1, len(vessel_points) // 50)
    centerline = vessel_points[::step]
  
    return centerline

def generate_trajectory_from_centerline(centerline, spacing=1.0):
    """从中心线生成轨迹"""
    if len(centerline) < 2:
        return []
  
    trajectory = []
    time_ms = 0
  
    for i in range(1, len(centerline)):
        # 计算推送距离
        prev_point = centerline[i-1] * spacing
        curr_point = centerline[i] * spacing
        push_distance = np.linalg.norm(curr_point - prev_point)
      
        # 计算旋转角度（简化：只考虑xy平面）
        dx = curr_point[0] - prev_point[0]
        dy = curr_point[1] - prev_point[1]
        angle = np.degrees(np.arctan2(dy, dx))
      
        # 设置速度
        push_velocity = min(5.0, push_distance * 2)
        angular_velocity = min(20.0, abs(angle))
      
        trajectory.append({
            'time_ms': time_ms,
            'push_mm': push_distance,
            'rotate_deg': angle,
            'velocity_mm_s': push_velocity,
            'angular_velocity_deg_s': angular_velocity
        })
      
        # 更新时间
        time_increment = max(
            push_distance / push_velocity * 1000,
            abs(angle) / angular_velocity * 1000
        )
        time_ms += int(time_increment)
  
    return trajectory

def save_trajectory_csv(trajectory, filename):
    """保存轨迹为CSV格式"""
    df = pd.DataFrame(trajectory)
  
    # 转换为累积值
    df['push_mm_cumsum'] = df['push_mm'].cumsum()
    df['rotate_deg_cumsum'] = df['rotate_deg'].cumsum()
  
    # 保存文件
    output_df = pd.DataFrame({
        'time_ms': df['time_ms'],
        'push_mm': df['push_mm_cumsum'],
        'rotate_deg': df['rotate_deg_cumsum'],
        'velocity_mm_s': df['velocity_mm_s'],
        'angular_velocity_deg_s': df['angular_velocity_deg_s']
    })
  
    output_df.to_csv(filename, index=False)
    print(f"轨迹已保存: {filename}")
    print(f"总时长: {df['time_ms'].iloc[-1] / 1000:.1f} 秒")
    print(f"总推送: {df['push_mm_cumsum'].iloc[-1]:.1f} mm")

# 生成测试轨迹
if __name__ == "__main__":
    # 创建简单的测试轨迹
    test_trajectory = [
        {'time_ms': 0, 'push_mm': 0, 'rotate_deg': 0, 
         'velocity_mm_s': 2, 'angular_velocity_deg_s': 10},
        {'time_ms': 1000, 'push_mm': 2, 'rotate_deg': 5, 
         'velocity_mm_s': 2, 'angular_velocity_deg_s': 10},
        {'time_ms': 2000, 'push_mm': 4, 'rotate_deg': 10, 
         'velocity_mm_s': 2, 'angular_velocity_deg_s': 10},
        {'time_ms': 3000, 'push_mm': 6, 'rotate_deg': 15, 
         'velocity_mm_s': 2, 'angular_velocity_deg_s': 10},
    ]
  
    # 保存轨迹
    output_dir = Path.home() / "surgical_robot_navigation" / "shared_data" / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)
  
    save_trajectory_csv(test_trajectory, output_dir / "ai_test_trajectory_001.csv")
```

### 验收标准

- [ ] nnU-Net训练启动成功
- [ ] 生成至少一个测试轨迹文件
- [ ] 轨迹文件格式验证通过

---

## 周末任务：整理和文档

### 任务1：创建推理脚本模板

创建 `inference_template.py`：

```python
"""推理脚本模板"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path
import torch

class VesselSegmentationInference:
    def __init__(self, model_path):
        self.model_path = model_path
        # TODO: 加载模型
      
    def preprocess(self, image_path):
        """预处理输入图像"""
        image = sitk.ReadImage(str(image_path))
        image_array = sitk.GetArrayFromImage(image)
      
        # 归一化
        image_array = (image_array - image_array.mean()) / image_array.std()
      
        # 添加batch和channel维度
        image_tensor = torch.from_numpy(image_array).float()
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
      
        return image_tensor, image
      
    def predict(self, image_tensor):
        """运行推理"""
        with torch.no_grad():
            # TODO: 实际推理代码
            # output = self.model(image_tensor)
            # prediction = torch.argmax(output, dim=1)
            pass
      
        return prediction
      
    def postprocess(self, prediction, original_image):
        """后处理预测结果"""
        # 转换回SimpleITK格式
        prediction_array = prediction.squeeze().cpu().numpy()
        prediction_image = sitk.GetImageFromArray(prediction_array.astype(np.uint8))
      
        # 复制原始图像信息
        prediction_image.CopyInformation(original_image)
      
        return prediction_image
      
    def segment_vessel(self, image_path):
        """完整的分割流程"""
        # 预处理
        image_tensor, original_image = self.preprocess(image_path)
      
        # 推理
        prediction = self.predict(image_tensor)
      
        # 后处理
        segmentation = self.postprocess(prediction, original_image)
      
        return segmentation

# 使用示例
if __name__ == "__main__":
    # 初始化
    segmenter = VesselSegmentationInference("path/to/model")
  
    # 分割
    result = segmenter.segment_vessel("path/to/image.nii.gz")
  
    # 保存结果
    sitk.WriteImage(result, "segmentation_result.nii.gz")
```

### 任务2：编写进度报告

创建 `week1_report_ai.md`：

```markdown
# AI组第一周进度报告

## 完成情况

### 1. 环境搭建
- [x] Anaconda环境配置
- [x] PyTorch安装（GPU/CPU）
- [x] MONAI框架安装
- [x] nnU-Net配置

### 2. 数据准备
- [x] 生成10个合成血管数据
- [x] 数据可视化验证
- [x] nnU-Net格式转换
- [x] 数据预处理完成

### 3. 模型训练
- [x] nnU-Net训练启动
- [ ] 完整模型训练（进行中）
- [x] 训练监控脚本

### 4. 轨迹生成
- [x] 简单轨迹生成测试
- [x] CSV格式输出
- [x] 与控制组接口对接

## 生成的测试轨迹
1. `ai_test_trajectory_001.csv` - 简单直线推进+旋转

## 遇到的问题
1. 问题：GPU显存不足
   解决：使用2D模型，减小batch size

2. 问题：真实DSA数据缺乏
   解决：先用合成数据验证流程

## 下周计划
1. 获取真实血管造影数据
2. 完成完整模型训练
3. 实现中心线提取算法
4. 优化轨迹生成逻辑
5. 创建推理API服务

## 需要协调事项
1. 确认轨迹文件格式是否满足控制组需求
2. 讨论实时推理的性能要求
3. 确定模型更新流程
```

### 任务3：环境备份脚本

```bash
#!/bin/bash
# backup_env.sh

echo "备份AI环境配置..."

# 导出conda环境
conda env export > environment.yml

# 导出pip包列表
pip freeze > requirements.txt

# 记录系统信息
echo "=== 系统信息 ===" > system_info.txt
echo "Python版本: $(python --version)" >> system_info.txt
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')" >> system_info.txt
echo "CUDA版本: $(nvcc --version | grep release)" >> system_info.txt
echo "GPU信息: $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> system_info.txt

echo "环境备份完成！"
```

---

## 每日检查清单

### 环境状态检查

```python
# check_ai_env.py
#!/usr/bin/env python3

import subprocess
import sys

def check_command(cmd, name):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {name}: {result.stdout.strip()}")
            return True
        else:
            print(f"✗ {name}: 错误")
            return False
    except:
        print(f"✗ {name}: 未找到")
        return False

print("=== AI组环境检查 ===")

# Python环境
check_command("python --version", "Python")
check_command("conda --version", "Conda")

# 深度学习框架
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
except:
    print("✗ PyTorch: 未安装")

try:
    import monai
    print(f"✓ MONAI: {monai.__version__}")
except:
    print("✗ MONAI: 未安装")

# nnU-Net
check_command("nnUNet_plan_and_preprocess -h > /dev/null 2>&1 && echo 'installed'", "nnU-Net")

# 数据目录
import os
data_dir = os.path.expanduser("~/vessel_dataset")
if os.path.exists(data_dir):
    n_files = len(os.listdir(data_dir))
    print(f"✓ 数据目录: {n_files} 个文件")
else:
    print("✗ 数据目录: 不存在")
```

---

## 技术资源链接

1. **MONAI教程**: https://monai.io/tutorials.html
2. **nnU-Net文档**: https://github.com/MIC-DKFZ/nnUNet
3. **医学图像处理**: https://simpleitk.readthedocs.io/
4. **深度学习资源**: https://pytorch.org/tutorials/

---

## 与控制组协作要点

1. **轨迹文件交付**

   - 位置：`~/surgical_robot_navigation/shared_data/trajectories/`
   - 命名：`ai_trajectory_XXX.csv`
   - 验证：使用共享的验证脚本
2. **性能指标**

   - 分割推理时间：< 100ms
   - 轨迹生成时间：< 50ms
   - 总处理时间：< 200ms
3. **接口规范**

   - 输入：DICOM/NIfTI格式图像
   - 输出：CSV格式轨迹文件

---

**第一周目标达成标准**：

- 环境搭建完成，所有依赖可用
- 至少10个标注数据准备就绪
- nnU-Net训练流程跑通
- 生成1-2个测试轨迹供控制组验证
- 建立基本的开发和测试流程