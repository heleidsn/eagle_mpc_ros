#!/usr/bin/env python
'''
Author: Lei He
Date: 2025-02-19 17:29:01
LastEditTime: 2025-03-10 19:43:55
Description: New version of MPC runner for PX4 SITL with eagle MPC
Github: https://github.com/heleidsn
'''
import rospy
from create_problem import get_opt_traj, create_mpc_controller
from eagle_mpc_msgs.msg import SolverPerformance, MpcState, MpcControl
import numpy as np
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State, AttitudeTarget
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Vector3
from threading import Lock


class MpcRunner:
    def __init__(self):
        
        # self.node_params = type('NodeParams', (), {})()
        # self.node_params.trajectory_config_path = rospy.get_param(f'{self.namespace}/trajectory_config', '')
        # rospy.loginfo(f"Loading trajectory config from: {self.node_params.trajectory_config_path}")
        
        robotName = "iris"
        trajectoryName = 'displacement_fix_yaw'
        self.dt_traj_opt = 5
        useSquash = True
        
        yaml_file_path = "/home/helei/catkin_eagle_mpc/src/eagle_mpc_ros/eagle_mpc_yaml"
        
        traj_solver, traj_state_ref, traj_problem, trajectory_obj = get_opt_traj(
            robotName, 
            trajectoryName, 
            self.dt_traj_opt, 
            useSquash,
            yaml_file_path)
        
        self.trajectory_duration = trajectory_obj.duration
        
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
        
        self.initialize_variables()
        
        rospy.loginfo("MPC controller initialized")
        
        self.odom_source = 'gazebo'
        
        # subscriber
        if self.odom_source == 'gazebo':
            self.odom_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback_model_state_gazebo, tcp_nodelay=True)
        elif self.odom_source == 'mavros':
            self.odom_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.callback_model_local_position)
        else:
            rospy.logerr(f"Invalid odom_source: {self.odom_source}")
            raise ValueError(f"Invalid odom_source: {self.odom_source}")
        
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.callback_state)
        
        # publisher
        self.control_cmd_pub = rospy.Publisher("/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10)
        self.mpc_state_pub = rospy.Publisher("/mpc/state", MpcState, queue_size=10)
        self.mpc_control_pub = rospy.Publisher("/mpc/control", MpcControl, queue_size=10)
        
        # Timer 1: MPC control
        self.controller_started = False 
        self.mpc_rate = 100.0  # Hz
        self.mpc_timer = rospy.Timer(rospy.Duration(1.0/self.mpc_rate), self.mpc_timer_callback)
        
        # timer 2: 1 Hz state check to start MPC controller
        self.mpc_status_timer = rospy.Timer(rospy.Duration(1), self.mpc_status_time_callback)

    def initialize_variables(self):
        if not hasattr(self, 'mpc_controller'):
            rospy.logerr("MPC controller not initialized!")
            raise RuntimeError("MPC controller not initialized!")

        # Initialize state variables
        self.state_lock = Lock()
        try:
            self.state = self.mpc_controller.state.zero()
        except AttributeError as e:
            rospy.logerr("Failed to get robot state: %s", str(e))
            raise
        self.state_ref = np.copy(self.state)
        self.state_diff = np.zeros(self.mpc_controller.state.ndx)

        # Initialize control variables
        self.control_command = np.zeros(self.mpc_controller.actuation.nu)
        # 初始化推力命令
        self.thrust_command = np.zeros(self.mpc_controller.platform_params.n_rotors)
        self.speed_command = np.zeros(self.mpc_controller.platform_params.n_rotors)
        self.total_thrust = 0.0

        # Initialize timing variables
        self.controller_time = rospy.Duration(0)
        self.controller_start_time = rospy.Time(0)
        self.control_last = rospy.Time(0)
        
        self.current_state = State()
        
        self.mpc_time_step = 0
        self.solving_time = 0.0

        rospy.loginfo("initializeVariables ok")

#region --------------command publisher--------------------------------

    def publish_mavros_rate_command(self):
        # using mavros setpoint to achieve rate control
        
        self.control_command = self.mpc_controller.solver.us_squash[0]
        self.thrust_command = self.control_command[:len(self.thrust_command)]
        
        # get planned state
        self.state_ref = self.mpc_controller.solver.xs[1]
        
        # get body rate command
        self.roll_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 3]
        self.pitch_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 4]
        self.yaw_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 5]
        
        # get thrust command
        self.total_thrust = np.sum(self.thrust_command)
        
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

        self.control_cmd_pub.publish(att_msg)
        
    def publish_mpc_debug_data(self):
        # 发布MPC状态
        state_msg = MpcState()
        state_msg.header.stamp = rospy.Time.now()
        state_msg.state = self.state.tolist()
        state_msg.state_ref = self.state_ref.tolist()
        state_msg.state_error = (self.state_ref - self.state).tolist()
        
        # 填充位置和姿态信息
        state_msg.position.x = self.state[0]
        state_msg.position.y = self.state[1]
        state_msg.position.z = -self.state[2]
        state_msg.orientation.x = self.state[3]
        state_msg.orientation.y = self.state[4]
        state_msg.orientation.z = self.state[5]
        state_msg.orientation.w = self.state[6]
        
        state_msg.mpc_time_step = self.mpc_ref_index
        state_msg.solving_time = self.solving_time
        
        # 发布控制输入
        control_msg = MpcControl()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.control_raw = self.mpc_controller.solver.us[0].tolist()
        control_msg.control_squash = self.mpc_controller.solver.us_squash[0].tolist()
        control_msg.thrust_command = self.mpc_controller.solver.xs[0].tolist()   # state 0 is current state
        control_msg.speed_command = self.mpc_controller.solver.xs[1].tolist()

        # 发布消息
        self.mpc_state_pub.publish(state_msg)
        self.mpc_control_pub.publish(control_msg)

#endregion

#region --------------msg callback--------------------------------

    def callback_model_state_gazebo(self, msg):
        """处理来自Gazebo的模型状态"""
        try:
            # 找到iris模型的索引
            idx = msg.name.index('iris')
            
            # 获取位置和姿态
            pose = msg.pose[idx]
            twist = msg.twist[idx]
            
            with self.state_lock:
                state_new = np.copy(self.state)
                # 更新位置
                state_new[0:3] = [pose.position.x,
                               pose.position.y,
                               pose.position.z]
                # 更新姿态四元数
                state_new[3:7] = [pose.orientation.x,
                               pose.orientation.y,
                               pose.orientation.z,
                               pose.orientation.w]
                
                nq = self.mpc_controller.state.nq
                # 更新线速度
                state_new[nq:nq+3] = [twist.linear.x,
                                    twist.linear.y,
                                    twist.linear.z]
                # 更新角速度
                state_new[nq+3:nq+6] = [twist.angular.x,
                                     twist.angular.y,
                                     twist.angular.z]
                
                self.state = state_new
                
            rospy.logdebug(f"Iris state: {self.state}")
            
        except ValueError:
            rospy.logwarn("Could not find iris model in gazebo model states")

    def callback_model_local_position(self, msg):
        """处理来自MAVROS的本地位置信息"""
        pose = msg.pose.pose
        twist = msg.twist.twist
        
        with self.state_lock:
            state_new = np.copy(self.state)
            # 更新位置
            state_new[0:3] = [pose.position.x,
                            pose.position.y,
                            pose.position.z]
            # 更新姿态四元数
            state_new[3:7] = [pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                            pose.orientation.w]
            
            nq = self.mpc_controller.state.nq
            # 更新线速度
            state_new[nq:nq+3] = [twist.linear.x,
                                twist.linear.y,
                                twist.linear.z]
            # 更新角速度
            state_new[nq+3:nq+6] = [twist.angular.x,
                                    twist.angular.y,
                                    twist.angular.z]
            
            self.state = state_new
    
    def callback_state(self, msg):
        """处理MAVROS状态回调"""
        self.current_state = msg

#endregion

#region --------------timer callback--------------------------------

    def mpc_timer_callback(self, event):
        # Set initial state and update problem
        self.mpc_controller.problem.x0 = self.state
            
        if self.controller_started:
            self.controller_time = rospy.Time.now() - self.controller_start_time
            self.mpc_ref_index = int(self.controller_time.to_sec() * 1000.0)
            
            # if self.mpc_ref_index >= 2000:
            #     self.mpc_ref_index = 2000
        else:
            self.mpc_ref_index = 0
            
        
        # update problem
        self.mpc_controller.updateProblem(self.mpc_ref_index)   # update problem using current time in ms
        
        time_start = rospy.Time.now()
        self.mpc_controller.solver.solve(
            self.mpc_controller.solver.xs,
            self.mpc_controller.solver.us,
            self.mpc_controller.iters
        )
        time_end = rospy.Time.now()
        self.solving_time = (time_end - time_start).to_sec()
        
        # rospy.loginfo("mpc_controller.solver.xs: %s", self.mpc_controller.solver.xs[0])
        # rospy.loginfo("mpc_controller.solver.us: %s", self.mpc_controller.solver.us_squash[0])

        # publish control command
        self.publish_mavros_rate_command()
        
        # publish mpc debug data
        self.publish_mpc_debug_data()
        
    def mpc_status_time_callback(self, event):
        # check if the controller is started
        if self.controller_started:
            rospy.loginfo("MPC controller is started")
        else:
            rospy.loginfo("MPC controller is not started")
            
        # check if model is offboard
        if self.current_state.mode == "OFFBOARD":
            rospy.loginfo("Model is offboard")
        else:
            rospy.loginfo("Model is not offboard")
            self.controller_started = False
        
        # check if model is armed
        if self.current_state.armed:
            rospy.loginfo("Model is armed")
        else:
            rospy.loginfo("Model is not armed")
            self.controller_started = False
            
        if not self.controller_started and self.current_state.mode == "OFFBOARD" and self.current_state.armed:
            rospy.loginfo("All conditions met for MPC start")
            self.controller_started = True
            self.controller_start_time = rospy.Time.now()
        else:
            # self.controller_started = False
            rospy.loginfo("Not all conditions met for MPC start")
        
#endregion


if __name__ == '__main__':
    rospy.init_node('mpc_runner')
    try:
        mpc_runner = MpcRunner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass