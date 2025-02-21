import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import os

# 设置 URDF 文件路径（需要替换为你的 URDF 文件路径）
# urdf_path = "/home/helei/catkin_eagle_mpc/src/example_robot_data/robots/iris_description/robots/iris.urdf"
urdf_path = "/home/helei/catkin_eagle_mpc/src/example_robot_data/robots/iris_description/robots/iris.urdf"

# 设置模型的 root 目录（通常是 URDF 文件所在目录的上级目录）
model_dir = os.path.dirname(urdf_path)

# 读取 URDF 文件
robot = RobotWrapper.BuildFromURDF(urdf_path, model_dir, pin.JointModelFreeFlyer())

# 获取模型和数据
model = robot.model
data = model.createData()

# 输出ndv
print(model.nv)

# 输出机器人信息
print("Robot name:", model.name)
print("Number of joints:", model.njoints)
print("Number of degrees of freedom (nv):", model.nv)
print("Configuration dimension (nq):", model.nq)

# 生成随机配置并计算前向运动学
q = pin.randomConfiguration(model)
pin.forwardKinematics(model, data, q)

# 读取末端位姿（假设机器人有一个名为 'end_effector' 的连杆）
# frame_id = model.getFrameId("end_effector")
# print("End-effector position:", data.oMf[frame_id].translation)
# print("End-effector rotation:", data.oMf[frame_id].rotation)