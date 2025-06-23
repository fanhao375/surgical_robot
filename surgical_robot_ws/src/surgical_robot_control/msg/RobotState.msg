# 机器人状态消息
float64 timestamp
float64 actual_push_position   # 实际推送位置(mm)
float64 actual_rotate_angle    # 实际旋转角度(度)
float64 push_force            # 推送力(N)
float64 rotate_torque         # 旋转力矩(Nm)
uint8 status                  # 0:空闲 1:运动中 2:错误
string error_message          # 错误信息 