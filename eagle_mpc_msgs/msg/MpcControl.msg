# MPC控制输入消息
Header header

float64[] control_raw     # 原始控制输入(solver.us)
float64[] control_squash  # 压缩后的控制输入(solver.us_squash)
float64[] thrust_command  # 推力命令
float64[] speed_command   # 电机速度命令