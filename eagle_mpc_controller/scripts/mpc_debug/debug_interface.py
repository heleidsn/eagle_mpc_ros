'''
Author: Lei He
Date: 2025-02-19 11:40:31
LastEditTime: 2025-02-21 09:41:43
Description: MPC Debug Interface, useful for debugging your MPC controller before deploying it to the real robot
Github: https://github.com/heleidsn
'''
#!/usr/bin/env python3

import rospy
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider
from eagle_mpc_msgs.msg import MpcState, MpcControl
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from std_msgs.msg import Float64

from utils.create_problem import get_opt_traj, create_mpc_controller

from mavros_msgs.msg import AttitudeTarget
from geometry_msgs.msg import Vector3

class MpcDebugInterface(QWidget):
    def __init__(self, using_ros=False):
        super(MpcDebugInterface, self).__init__()
        self.setWindowTitle('MPC Debug Interface')
        
        # 创建布局
        self.layout = QVBoxLayout()
        
        # 时间控制
        time_layout = QHBoxLayout()
        self.time_slider = QSlider(QtCore.Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(2000)  # 2秒
        self.time_slider.valueChanged.connect(self.time_changed)
        self.time_label = QLabel('0 ms')
        time_layout.addWidget(QLabel('Time (ms):'))
        time_layout.addWidget(self.time_slider)
        time_layout.addWidget(self.time_label)
        
        # 状态修改
        state_layout = QHBoxLayout()
        self.state_sliders = {}
        self.state_labels = {}
        
        # 创建滑块和标签
        slider_configs = [
            ('X', -2, 2),
            ('Y', -2, 2),
            ('Z', 0, 4)
        ]
        
        for name, min_val, max_val in slider_configs:
            slider_layout = self.create_state_slider(name, min_val, max_val)
            state_layout.addLayout(slider_layout)
        
        # 图表显示
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(211)
        self.ax_rate = self.figure.add_subplot(212)
        
        # 添加到主布局
        self.layout.addLayout(time_layout)
        self.layout.addLayout(state_layout)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)
      
        # 数据存储
        self.state_history = []
        self.control_history = []
        
        # 定时更新图表
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)  # 10Hz更新
        
        # initialize MPC
        
        robotName = 'iris'
        trajectoryName = 'hover'
        self.dt_traj_opt = 20
        useSquash = True
        
        yaml_file_path = "/home/helei/catkin_eagle_mpc/src/eagle_mpc_ros/eagle_mpc_yaml"
        
        traj_solver, traj_state_ref, traj_problem, trajectory_obj = get_opt_traj(
            robotName, 
            trajectoryName, 
            self.dt_traj_opt, 
            useSquash,
            yaml_file_path)
        
        # create mpc controller to get tau_f
        mpc_name = "rail"
        mpc_yaml = '{}/mpc/{}_mpc.yaml'.format(yaml_file_path, robotName)
        self.mpc_controller = create_mpc_controller(
            mpc_name,
            trajectory_obj,
            traj_state_ref,
            self.dt_traj_opt,
            mpc_yaml
        )
        
        self.state = self.mpc_controller.state.zero()
        self.state_ref = np.copy(self.mpc_controller.state_ref)
        self.mpc_ref_index = 0
        self.solving_time = 0.0
        
        rospy.loginfo("MPC controller initialized")
        
        # ROS订阅和发布
        self.pose_pub = rospy.Publisher('/debug/pose', PoseStamped, queue_size=1)
        self.time_pub = rospy.Publisher('/debug/time', Float64, queue_size=1)
        
        self.state_sub = rospy.Subscriber('/debug/mpc_state', MpcState, self.state_callback)
        self.control_sub = rospy.Subscriber('/debug/mpc_control', MpcControl, self.control_callback)
        
        
        self.mpc_state_pub = rospy.Publisher('/mpc/state', MpcState, queue_size=10)
        self.mpc_control_pub = rospy.Publisher('/mpc/control', MpcControl, queue_size=10)
        
        self.solving_time_pub = rospy.Publisher('/mpc/solving_time', Float64, queue_size=1)
        
        self.attitude_pub = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)
        
        # start MPC controller and wait for it to start
        self.mpc_rate = 10.0  # Hz
        self.mpc_timer = rospy.Timer(rospy.Duration(1.0/self.mpc_rate), self.mpc_timer_callback)
        
        rospy.loginfo(f"MPC started at {self.mpc_rate}Hz")
        
    def mpc_timer_callback(self, event):
        
        self.mpc_controller.problem.x0 = self.state
        
        print(self.mpc_ref_index)
        self.mpc_controller.updateProblem(self.mpc_ref_index)
        
        time_start = rospy.Time.now()
        self.mpc_controller.solver.solve(
            self.mpc_controller.solver.xs,
            self.mpc_controller.solver.us,
            self.mpc_controller.iters
        )
        time_end = rospy.Time.now()

        self.solving_time = (time_end - time_start).to_sec()
        
        # 发布MPC数据
        # self.publish_mpc_data()
        
        # self.publish_mavros_rate_command()
        
    def publish_mavros_rate_command(self):
        # using mavros setpoint to achieve rate control
        
        self.control_command = self.mpc_controller.solver.us_squash[0]
        
        # get planned state
        self.state_ref = self.mpc_controller.solver.xs[1]
        
        self.roll_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 3]
        self.pitch_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 4]
        self.yaw_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 5]
        
        self.total_thrust = np.sum(self.control_command)
        
        att_msg = AttitudeTarget()
        att_msg.header.stamp = rospy.Time.now()
        
        # 设置 type_mask，忽略姿态，仅使用角速度 + 推力
        att_msg.type_mask = AttitudeTarget.IGNORE_ATTITUDE 
        
        # 机体系角速度 (rad/s)
        att_msg.body_rate = Vector3(self.roll_rate_ref, self.pitch_rate_ref, self.yaw_rate_ref)  # 仅绕 Z 轴旋转 0.1 rad/s
        
        # 推力值 (范围 0 ~ 1)
        max_thrust = 7.0664 * 4
        att_msg.thrust = self.total_thrust / max_thrust  # 60% 油门
        
        # 对推力进行限幅
        att_msg.thrust = np.clip(att_msg.thrust, 0, 1)

        self.attitude_pub.publish(att_msg)
        
    def publish_mpc_data(self):
        # 发布MPC状态
        state_msg = MpcState()
        state_msg.header.stamp = rospy.Time.now()
        state_msg.state = self.state.tolist()
        state_msg.state_ref = self.state_ref.tolist()
        state_msg.state_error = (self.state_ref - self.state).tolist()
        
        # 填充位置和姿态信息
        state_msg.position.x = self.state[0]
        state_msg.position.y = self.state[1]
        state_msg.position.z = self.state[2]
        state_msg.orientation.x = self.state[3]
        state_msg.orientation.y = self.state[4]
        state_msg.orientation.z = self.state[5]
        state_msg.orientation.w = self.state[6]
        
        # 发布控制输入
        control_msg = MpcControl()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.control_raw = self.mpc_controller.solver.us[0].tolist()
        control_msg.control_squash = self.mpc_controller.solver.us_squash[0].tolist()
        control_msg.thrust_command = self.mpc_controller.solver.xs[0].tolist()
        control_msg.speed_command = self.mpc_controller.solver.xs[1].tolist()
        
        # 发布消息
        self.mpc_state_pub.publish(state_msg)
        self.mpc_control_pub.publish(control_msg)
        self.solving_time_pub.publish(Float64(self.solving_time))
        
    def create_state_slider(self, name, min_val, max_val):
        layout = QVBoxLayout()
        
        # 标题标签
        title_label = QLabel(name)
        layout.addWidget(title_label)
        
        # 滑块
        slider = QSlider(QtCore.Qt.Vertical)
        slider.setMinimum(int(min_val * 100))
        slider.setMaximum(int(max_val * 100))
        slider.setValue(0)  # 设置初始值
        slider.valueChanged.connect(lambda: self.state_changed(name, slider.value()/100.0))
        layout.addWidget(slider)
        
        # 值标签
        value_label = QLabel('0.00')
        value_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(value_label)
        
        # 存储滑块和标签的引用
        self.state_sliders[name] = slider
        self.state_labels[name] = value_label
        
        return layout
        
    def time_changed(self, value):
        # 发布新的时间戳
        self.time_pub.publish(Float64(value))
        self.time_label.setText(f'{value} ms')
        
        self.mpc_ref_index = int(value / self.dt_traj_opt)
        
        # contstrain the time to be within the range of the trajectory
        print(self.mpc_ref_index, len(self.state_ref))
        
        if self.mpc_ref_index < 0:
            self.mpc_ref_index = 0
        elif self.mpc_ref_index > len(self.state_ref):
            self.mpc_ref_index = len(self.state_ref)
        
        
    def state_changed(self, axis, value):
        # 发布新的位置
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "world"
        if axis == 'X':
            pose.pose.position.x = value
            self.state[0] = value
        elif axis == 'Y':
            pose.pose.position.y = value
            self.state[1] = value
        elif axis == 'Z':
            pose.pose.position.z = value
            self.state[2] = value
        
        self.pose_pub.publish(pose)
        
        # 更新显示的值
        self.state_labels[axis].setText(f'{value:.2f}')
        
    def state_callback(self, msg):
        self.state_history.append(msg)
        if len(self.state_history) > 100:
            self.state_history.pop(0)
            
        # 更新滑块位置以匹配当前状态
        if self.state_history:
            current_state = self.state_history[-1].state
            for i, axis in enumerate(['X', 'Y', 'Z']):
                self.state_sliders[axis].blockSignals(True)  # 防止触发回调
                self.state_sliders[axis].setValue(int(current_state[i] * 100))
                self.state_labels[axis].setText(f'{current_state[i]:.2f}')
                self.state_sliders[axis].blockSignals(False)
            
    def control_callback(self, msg):
        self.control_history.append(msg)
        if len(self.control_history) > 100:
            self.control_history.pop(0)
            
    def update_plot(self):
        
        
        state_predict = np.array(self.mpc_controller.solver.xs)
        state_ref = np.array(self.mpc_controller.state_ref)
        
        self.ax.clear()
        
        self.ax.set_title('MPC Debug Interface')
        self.ax.set_xlabel('Time (ms)')
        self.ax.set_ylabel('State')
        
        self.ax.plot(state_predict[:, 0], label='X')
        self.ax.plot(state_predict[:, 1], label='Y')
        self.ax.plot(state_predict[:, 2], label='Z')
        self.ax.plot(state_ref[:, 0], label='X_ref')
        self.ax.plot(state_ref[:, 1], label='Y_ref')
        self.ax.plot(state_ref[:, 2], label='Z_ref')
        
        self.ax.legend()
        self.canvas.draw()
        
        # 创建subplot for rate target
        # self.ax_rate = self.figure.add_subplot(122)
        # self.ax_rate.clear()
        # self.ax_rate.set_title('Rate Target')   
        # self.ax_rate.set_xlabel('Time (ms)')
        # self.ax_rate.set_ylabel('Rate')
        
        # self.ax_rate.plot(state_predict[:, -3], label='Roll Rate')
        # self.ax_rate.plot(state_predict[:, -2], label='Pitch Rate')
        # self.ax_rate.plot(state_predict[:, -1], label='Yaw Rate')
        
        # self.ax_rate.plot(state_ref[:, -3], label='Roll Rate SP')
        # self.ax_rate.plot(state_ref[:, -2], label='Pitch Rate SP')
        # self.ax_rate.plot(state_ref[:, -1], label='Yaw Rate SP')
        
        # self.ax_rate.legend()
        # self.canvas.draw()


if __name__ == '__main__':
    import sys
    rospy.init_node('mpc_debug_interface')
    
    app = QtWidgets.QApplication(sys.argv)
    window = MpcDebugInterface()
    window.show()
    
    sys.exit(app.exec_()) 