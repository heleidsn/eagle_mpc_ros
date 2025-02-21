# MPC状态消息
Header header

# 状态向量
float64[] state           # 完整状态向量
float64[] state_ref      # 参考状态向量
float64[] state_error    # 状态误差

int32 mpc_time_step      # 迭代位置
float64 solving_time     # 求解时间

# 位置和姿态
geometry_msgs/Point position
geometry_msgs/Quaternion orientation
geometry_msgs/Vector3 velocity
geometry_msgs/Vector3 angular_velocity 