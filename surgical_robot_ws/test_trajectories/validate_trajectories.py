#!/usr/bin/env python3
"""
轨迹文件验证脚本
验证CSV格式和数据有效性
"""

import csv
import os
import sys

def validate_trajectory_file(filename):
    """验证单个轨迹文件"""
    print(f"\n=== 验证文件: {filename} ===")
    
    if not os.path.exists(filename):
        print(f"❌ 文件不存在: {filename}")
        return False
    
    try:
        with open(filename, 'r') as file:
            reader = csv.DictReader(file)
            
            # 检查必需的列
            required_columns = [
                'time_ms', 'push_mm', 'rotate_deg', 
                'velocity_mm_s', 'angular_velocity_deg_s'
            ]
            
            if reader.fieldnames is None or not all(col in reader.fieldnames for col in required_columns):
                print(f"❌ 缺少必需的列。需要: {required_columns}")
                print(f"    实际列: {reader.fieldnames}")
                return False
            
            print(f"✅ 列格式正确: {reader.fieldnames}")
            
            # 验证数据行
            rows = list(reader)
            if len(rows) == 0:
                print("❌ 文件中没有数据行")
                return False
            
            print(f"✅ 数据行数: {len(rows)}")
            
            # 验证数据类型和范围
            prev_time = -1
            for i, row in enumerate(rows):
                try:
                    time_ms = float(row['time_ms'])
                    push_mm = float(row['push_mm'])
                    rotate_deg = float(row['rotate_deg'])
                    velocity_mm_s = float(row['velocity_mm_s'])
                    angular_velocity_deg_s = float(row['angular_velocity_deg_s'])
                    
                    # 检查时间戳递增
                    if time_ms <= prev_time:
                        print(f"❌ 第{i+1}行: 时间戳未递增 ({time_ms} <= {prev_time})")
                        return False
                    prev_time = time_ms
                    
                    # 检查合理范围
                    if abs(push_mm) > 1000:  # 假设最大推送1000mm
                        print(f"⚠️  第{i+1}行: 推送位置超出合理范围: {push_mm}mm")
                    
                    if abs(rotate_deg) > 720:  # 假设最大旋转720度
                        print(f"⚠️  第{i+1}行: 旋转角度超出合理范围: {rotate_deg}°")
                    
                    if abs(velocity_mm_s) > 100:  # 假设最大速度100mm/s
                        print(f"⚠️  第{i+1}行: 推送速度超出合理范围: {velocity_mm_s}mm/s")
                    
                    if abs(angular_velocity_deg_s) > 360:  # 假设最大角速度360°/s
                        print(f"⚠️  第{i+1}行: 角速度超出合理范围: {angular_velocity_deg_s}°/s")
                
                except ValueError as e:
                    print(f"❌ 第{i+1}行: 数据类型错误 - {e}")
                    return False
            
            # 打印轨迹摘要
            total_time = float(rows[-1]['time_ms']) / 1000.0  # 转换为秒
            final_push = float(rows[-1]['push_mm'])
            final_rotate = float(rows[-1]['rotate_deg'])
            
            print(f"📊 轨迹摘要:")
            print(f"    总时长: {total_time:.1f}秒")
            print(f"    最终推送位置: {final_push:.1f}mm")
            print(f"    最终旋转角度: {final_rotate:.1f}°")
            print(f"✅ 文件验证通过")
            
            return True
            
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False

def main():
    """主函数"""
    print("🔍 轨迹文件验证工具")
    print("=" * 50)
    
    # 获取当前目录下的所有CSV文件
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ 当前目录下没有找到CSV文件")
        return False
    
    print(f"找到 {len(csv_files)} 个CSV文件: {csv_files}")
    
    all_valid = True
    for csv_file in sorted(csv_files):
        valid = validate_trajectory_file(csv_file)
        all_valid &= valid
    
    print("\n" + "=" * 50)
    if all_valid:
        print("🎉 所有轨迹文件验证通过！")
        print("✅ 可以用于Day4的轨迹播放测试")
    else:
        print("❌ 部分文件验证失败，请检查并修正")
    
    return all_valid

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 