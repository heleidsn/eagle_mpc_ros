#!/usr/bin/env python

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import JointState
from mav_msgs.msg import Actuators
from mav_msgs.msg import RollPitchYawrateThrust, RateThrust
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64
from mavros_msgs.msg import AttitudeTarget
from eagle_mpc_msgs.msg import SolverPerformance, MpcState, MpcControl
from dynamic_reconfigure.server import Server
from eagle_mpc_controller.cfg import ParamsConfig
import eagle_mpc
import crocoddyl
from threading import Lock
import time
from scipy.spatial.transform import Rotation as R
from mavros_msgs.srv import SetMode, CommandBool
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped

np.set_printoptions(precision=4, suppress=True)  # suppress=True 禁止科学计数法

class MpcRunner:
    def __init__(self):
        # Get namespace before initializing parameters
        self.namespace = rospy.get_namespace()
        if self.namespace == '/':
            # If no namespace is set, try to get it from node name
            node_name = rospy.get_name()
            if '/' in node_name:
                self.namespace = node_name.rsplit('/', 1)[0] + '/'
            else:
                # Default to Carrot if no namespace is found
                self.namespace = '/Carrot/'
                
        self.namespace = '/Carrot'
        rospy.loginfo(f"Using namespace: {self.namespace}")
        
        self.solving_time = 0.0
        self.mpc_time_step = 0

        self.initialize_parameters()
        self.initialize_mpc_controller()
        self.initialize_variables()
        self.initialize_msgs()
        self.initialize_subscribers()

        rospy.loginfo("MpcRunner init ok")


        self.controller_started = False
        self.initialize_publishers()
        
        # start MPC controller and wait for it to start
        self.mpc_rate = 20.0  # Hz
        self.mpc_timer = rospy.Timer(rospy.Duration(1.0/self.mpc_rate), self.mpc_timer_callback)
        
        rospy.loginfo(f"MPC started at {self.mpc_rate}Hz")
        
        self.mpc_status_timer = rospy.Timer(rospy.Duration(1), self.mpc_status_time_callback)
        
        # try:
        #     # RQT Config
        #     self.server = Server(ParamsConfig, self.callback_config)
        #     self.node_params.initialized_time = rospy.Time.now()
        # except Exception as e:
        #     rospy.logerr("Failed to initialize dynamic reconfigure server: %s", str(e))
        #     raise

    def initialize_parameters(self):
        # Create a namespace for parameters
        self.node_params = type('NodeParams', (), {})()
        
        # 从ROS参数服务器获取倒计时时间，默认为5秒
        self.node_params.start_seconds = rospy.get_param(f'{self.namespace}/start_countdown', 1)
        
        # Trajectory related
        self.node_params.trajectory_config_path = rospy.get_param(f'{self.namespace}/trajectory_config', '')
        rospy.loginfo(f"Loading trajectory config from: {self.node_params.trajectory_config_path}")
        
        trajectory_dt = rospy.get_param(f'{self.namespace}/trajectory_dt', 10)
        self.node_params.trajectory_dt = trajectory_dt

        trajectory_solver = rospy.get_param(f'{self.namespace}/trajectory_solver', 'SolverSbFDDP')
        if trajectory_solver == 'SolverSbFDDP':
            self.node_params.trajectory_solver = eagle_mpc.SolverSbFDDP
        else:
            self.node_params.trajectory_solver = eagle_mpc.SolverBoxFDDP

        self.node_params.trajectory_integration = rospy.get_param(f'{self.namespace}/trajectory_integration', 'IntegratedActionModelEuler')
        self.node_params.trajectory_squash = self.node_params.trajectory_solver == eagle_mpc.SolverSbFDDP

        self.node_params.mpc_config_path = rospy.get_param(f'{self.namespace}/mpc_config', '')
        self.node_params.mpc_type = rospy.get_param(f'{self.namespace}/mpc_type', '')
        self.node_params.use_internal_gains = rospy.get_param(f'{self.namespace}/use_internal_gains', False)
        self.node_params.record_solver = rospy.get_param(f'{self.namespace}/record_solver', False)
        self.node_params.record_solver_level = rospy.get_param(f'{self.namespace}record_solver_level', 0)
        self.node_params.automatic_start = rospy.get_param(f'{self.namespace}/automatic_start', False)
        self.node_params.arm_name = rospy.get_param(f'{self.namespace}/arm_name', '')
        self.node_params.arm_enable = bool(self.node_params.arm_name)
        self.node_params.use_roll_pitch_yawrate_thrust_control = rospy.get_param(f'{self.namespace}/use_roll_pitch_yawrate_thrust_control', False)
        self.node_params.motor_command_dt = rospy.get_param(f'{self.namespace}/motor_command_dt', 0)

        rospy.loginfo("initializeParameters ok")

    def initialize_mpc_controller(self):
        rospy.logwarn("initializeMpcController start")
        
        # 打印更多调试信息
        rospy.loginfo("Loading trajectory config from: %s", self.node_params.trajectory_config_path)
        
        self.trajectory = eagle_mpc.Trajectory()  # 直接创建实例
        
        try:
            print(self.node_params.trajectory_config_path)
            self.trajectory.autoSetup(self.node_params.trajectory_config_path)
        except RuntimeError as e:
            rospy.logerr("Failed to setup trajectory: %s", str(e))
            raise

        rospy.logwarn("initializeMpcController ok")

        # self.node_params.trajectory_dt = 10
        problem = self.trajectory.createProblem(
            self.node_params.trajectory_dt,
            self.node_params.trajectory_squash,
            self.node_params.trajectory_integration
        )

        if self.node_params.trajectory_solver == eagle_mpc.SolverSbFDDP:
            solver = eagle_mpc.SolverSbFDDP(problem, self.trajectory.squash)
        else:
            solver = crocoddyl.SolverBoxFDDP(problem)

        callbacks = [crocoddyl.CallbackVerbose()]
        solver.setCallbacks(callbacks)
        solver.solve()

        try:
            # Initialize MPC controller based on type
            if self.node_params.mpc_type == "Carrot":
                self.mpc_controller = eagle_mpc.CarrotMpc(
                    self.trajectory,
                    solver.xs,
                    self.node_params.trajectory_dt,
                    self.node_params.mpc_config_path
                )
            elif self.node_params.mpc_type == "Rail":
                self.mpc_controller = eagle_mpc.RailMpc(
                    solver.xs,
                    self.node_params.trajectory_dt,
                    self.node_params.mpc_config_path
                )
            elif self.node_params.mpc_type == "Weighted":
                self.mpc_controller = eagle_mpc.WeightedMpc(
                    self.trajectory,
                    self.node_params.trajectory_dt,
                    self.node_params.mpc_config_path
                )
            else:
                rospy.logerr("Unknown MPC type: %s", self.node_params.mpc_type)
                raise ValueError(f"Unknown MPC type: {self.node_params.mpc_type}")
            
            # self.mpc_controller.solver.setCallbacks([crocoddyl.CallbackVerbose()])  # 设置回调函数 
        except Exception as e:
            rospy.logerr("Failed to create MPC controller: %s", str(e))
            raise

        rospy.loginfo("initializeMpcController ok")

    def initialize_variables(self):
        if not hasattr(self, 'mpc_controller'):
            rospy.logerr("MPC controller not initialized!")
            raise RuntimeError("MPC controller not initialized!")

        # 等待MAVROS服务
        rospy.loginfo("Waiting for MAVROS services...")
        rospy.wait_for_service('/mavros/cmd/arming')
        rospy.wait_for_service('/mavros/set_mode')
        self.arming_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        
        # # 等待FCU连接
        # while not rospy.is_shutdown() and (not hasattr(self, 'current_state') or not self.current_state.connected):
        #     rospy.loginfo_throttle(1, "Waiting for FCU connection...")
        #     rospy.sleep(0.1)
        
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
        self.thrust_command = np.zeros(self.mpc_controller.platform_params.n_rotors)
        

        self.speed_command = self.thrust_command.copy()

        # Initialize timing variables
        self.controller_time = rospy.Duration(0)
        self.controller_start_time = rospy.Time(0)
        self.control_last = rospy.Time(0)

        rospy.loginfo("initializeVariables ok")

    def initialize_msgs(self):
        self.msg_thrusts = Actuators()
        self.msg_thrusts.angular_velocities = [0.0] * len(self.thrust_command)
        
        self.msg_roll_pitch_yawrate_thrust = RollPitchYawrateThrust()
        self.msg_roll_pitch_yawrate_thrust.header.stamp = rospy.Time.now()
        
        self.msg_roll_pitch_yawrate_thrust.roll = 0.0
        self.msg_roll_pitch_yawrate_thrust.pitch = 0.0
        self.msg_roll_pitch_yawrate_thrust.yaw_rate = 0.0
        # self.msg_roll_pitch_yawrate_thrust.thrust.thrust = 0.0
        
        if self.node_params.record_solver:
            self.msg_solver_performance = SolverPerformance()
            if self.node_params.record_solver_level > 0:
                self.msg_solver_performance.floating_base_trajectory = [None] * self.mpc_controller.get_knots()

    def initialize_subscribers(self):
        # choose odometry source
        self.odom_source = rospy.get_param(f'{self.namespace}/odom_source', 'gazebo')
        self.odom_source = 'gazebo'
        
        # 订阅mavros状态
        self.current_state = State()
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.callback_state)
        
        if self.odom_source == 'mavros':
            self.odom_sub = rospy.Subscriber(
                "/mavros/odometry/in", Odometry, self.callback_model_state_mavros, tcp_nodelay=True)
        elif self.odom_source == 'gazebo':
            self.model_state_sub = rospy.Subscriber(
                "/gazebo/model_states", ModelStates, self.callback_model_state_gazebo, tcp_nodelay=True)
        else:
            rospy.logerr(f"Invalid odom_source: {self.odom_source}")
            raise ValueError(f"Invalid odom_source: {self.odom_source}")
        
        if self.node_params.arm_enable:
            self.joint_state_sub = rospy.Subscriber(
                "/hexacopter370/joint_states", JointState, self.callback_joint_state, tcp_nodelay=True)


    def initialize_publishers(self):
        self.mpc_state_pub = rospy.Publisher('/mpc/state', MpcState, queue_size=10)
        self.mpc_control_pub = rospy.Publisher('/mpc/control', MpcControl, queue_size=10)
        
        if self.node_params.record_solver:
            self.solver_performance_pub = rospy.Publisher(
                "/mpc/solver_performance", SolverPerformance, queue_size=1)
        
        self.thrust_pub = rospy.Publisher("/hexacopter370/command/motor_speed", Actuators, queue_size=1)
        self.roll_pitch_yawrate_pub = rospy.Publisher("/hexacopter370/command/roll_pitch_yawrate_thrust", RollPitchYawrateThrust, queue_size=1)
        
        self.attitude_pub = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)

        if self.node_params.arm_enable:
            n_joints = self.mpc_controller.robot_model.nq - 7
            self.arm_pubs = []
            for i in range(n_joints):
                self.arm_pubs.append(
                    rospy.Publisher(f"/hexacopter370/joint{i+1}_effort_controller/command", Float64, queue_size=1))

    def mpc_status_time_callback(self, event):
        rospy.loginfo(f"MPC status time: {self.controller_time.to_sec()}")
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
        
        # check if model is armed
        if self.current_state.armed:
            rospy.loginfo("Model is armed")
        else:
            rospy.loginfo("Model is not armed")
            
        if not self.controller_started and self.current_state.mode == "OFFBOARD" and self.current_state.armed:
            rospy.loginfo("All conditions met for MPC start")
            self.controller_started = True
            self.controller_start_time = rospy.Time.now()
        else:
            rospy.loginfo("Not all conditions met for MPC start")

    def callback_model_state_mavros(self, msg):
        # update state
        with self.state_lock:
            # Create temporary array and assign values
            state_new = np.copy(self.state)
            if self.odom_source == 'mavros':
                state_new[0:3] = [msg.pose.pose.position.x,
                                msg.pose.pose.position.y,
                                -msg.pose.pose.position.z]
            elif self.odom_source == 'gazebo':
                state_new[0:3] = [msg.pose.pose.position.x,
                                msg.pose.pose.position.y,
                                msg.pose.pose.position.z]
            else:
                rospy.logerr(f"Invalid odom_source: {self.odom_source}")
                raise ValueError(f"Invalid odom_source: {self.odom_source}")
            
            state_new[3:7] = [msg.pose.pose.orientation.x,
                           msg.pose.pose.orientation.y,
                           msg.pose.pose.orientation.z,
                           msg.pose.pose.orientation.w]
            nq = self.mpc_controller.state.nq
            state_new[nq:nq+3] = [msg.twist.twist.linear.x,
                               msg.twist.twist.linear.y,
                               msg.twist.twist.linear.z]
            state_new[nq+3:nq+6] = [msg.twist.twist.angular.x,
                                 msg.twist.twist.angular.y,
                                 msg.twist.twist.angular.z]
            self.state = state_new

    def mpc_timer_callback(self, event):
        """100Hz定时器回调函数，执行MPC迭代"""
        self.run_mpc_iteration()

    def run_mpc_iteration(self, msg=None):      
        # Set initial state and update problem
        with self.state_lock:
            self.mpc_controller.problem.x0 = self.state
            
        if self.controller_started:
            self.controller_time = rospy.Time.now() - self.controller_start_time
            controller_instant = int(self.controller_time.to_sec() * 1000.0 / self.node_params.trajectory_dt)
        else:
            controller_instant = 0
            
            self.mpc_time_step = controller_instant
            print(f"mpc_time_step: {self.mpc_time_step}")
        # update problem
        self.mpc_controller.updateProblem(controller_instant)
        

        # Solve MPC problem
        if self.node_params.record_solver:
            self.solver_time_init = time.time()
        
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

        # if self.controller_started: 
            # Get control commands and publish
        self.publish_mavros_rate_command()
        
        if self.node_params.record_solver:
            self.publish_solver_performance(msg)
            
        # 发布MPC数据
        self.publish_mpc_data()

    def publish_mavros_rate_command(self):
        # using mavros setpoint to achieve rate control
        
        self.control_command = self.mpc_controller.solver.us_squash[0]
        self.thrust_command = self.control_command[:len(self.thrust_command)]
        self.speed_command = np.sqrt(self.thrust_command / self.mpc_controller.platform_params.cf)
        
        # get planned state
        self.state_ref = self.mpc_controller.solver.xs[1]
        
        self.roll_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 3]
        self.pitch_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 4]
        self.yaw_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 5]
        
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

        self.attitude_pub.publish(att_msg)

    def publish_commands(self):
        # Get control commands
        self.control_command = self.mpc_controller.solver.us_squash[0]
        self.thrust_command = self.control_command[:len(self.thrust_command)]
        self.speed_command = np.sqrt(self.thrust_command / self.mpc_controller.platform_params.cf)

        # Publish thrust commands
        self.msg_thrusts.header.stamp = rospy.Time.now()
        self.msg_thrusts.angular_velocities = self.speed_command.tolist()
        self.thrust_pub.publish(self.msg_thrusts)

        # Publish joint commands if arm is enabled
        if self.node_params.arm_enable:
            n_joints = self.mpc_controller.robot_model.nq - 7
            for i, pub in enumerate(self.arm_pubs):
                msg = Float64()
                msg.data = self.control_command[len(self.speed_command) + i]
                pub.publish(msg)

    def publish_commands_rate(self):
        # get control command
        self.control_command = self.mpc_controller.solver.us_squash[0]
        self.thrust_command = self.control_command[:len(self.thrust_command)]
        self.speed_command = np.sqrt(self.thrust_command / self.mpc_controller.platform_params.cf)
        
        # get planned state
        self.state_ref = self.mpc_controller.solver.xs[0]
        
        # get total thrust and angular velocities
        self.total_thrust = np.sum(self.thrust_command)
        
        # get quaternion from state_ref
        self.quaternion = self.state_ref[3:7]
        
        # 创建旋转对象
        rotation = R.from_quat(self.quaternion)
        euler_angles = rotation.as_euler('xyz', degrees=True)  # 使用 'xyz' 表示顺序，degrees=True 表示以度为单位
        
        # transform quaternion to roll, pitch, yaw
        self.roll_ref = euler_angles[0]
        self.pitch_ref = euler_angles[1]
        self.yaw_ref = euler_angles[2]

        
        # get roll, pitch, yawrate from state_ref
        
        self.roll_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 3]
        self.pitch_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 4]
        self.yaw_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 5]
        
        # publish roll, pitch, yawrate
        self.msg_roll_pitch_yawrate_thrust.roll = self.roll_ref
        self.msg_roll_pitch_yawrate_thrust.pitch = self.pitch_ref
        self.msg_roll_pitch_yawrate_thrust.yaw_rate = self.yaw_rate_ref
        self.msg_roll_pitch_yawrate_thrust.thrust.z = self.total_thrust
        
        self.roll_pitch_yawrate_pub.publish(self.msg_roll_pitch_yawrate_thrust)
        # print(self.msg_roll_pitch_yawrate_thrust)
        
    def set_mode(self, mode):
        rospy.wait_for_service('/mavros/set_mode')
        try:
            set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)
            response = set_mode_client(custom_mode=mode)
            return response.mode_sent
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to set mode: {e}")
            return False

    def arm(self):
        rospy.wait_for_service('/mavros/cmd/arming')
        try:
            arm_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
            response = arm_client(value=True)
            return response.success
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to arm: {e}")
            return False

    def callback_config(self, config, level):
        rospy.loginfo("RECONFIGURE_REQUEST!")
        self.controller_started = config.start_mission
        
        # if self.controller_started:
        #     # 启动准备线程
        #     rospy.Timer(rospy.Duration(0.1), self.prepare_flight, oneshot=True)
        # else:
        #     # 停止MPC定时器
        #     if self.mpc_timer is not None:
        #         self.mpc_timer.shutdown()
        #         self.mpc_timer = None
        #     rospy.loginfo("Stopped MPC timer")
        return config

    def prepare_flight(self, event):
        """准备飞行的线程函数"""
        try:
            # 1. 确保FCU已连接
            if not self.current_state.connected:
                rospy.logerr("FCU not connected")
                self.controller_started = False
                return
            
            # 2. 尝试解锁
            retry_count = 0
            while not rospy.is_shutdown() and not self.current_state.armed:
                if retry_count >= 5:
                    rospy.logerr("Failed to arm after 5 attempts")
                    self.controller_started = False
                    return
                
                if self.arming_client.call(True).success:
                    rospy.loginfo("Vehicle armed")
                else:
                    rospy.logwarn("Arming failed, retrying...")
                    retry_count += 1
                rospy.sleep(1)
            
            # 3. 切换到OFFBOARD模式
            # 需要先发送一些空的offboard 控制信号
            for i in range(100):
                if(rospy.is_shutdown()):
                    break

                self.publish_mavros_rate_command()
                rospy.sleep(0.01)
            
            retry_count = 0
            while not rospy.is_shutdown() and self.current_state.mode != "OFFBOARD":
                if retry_count >= 5:
                    rospy.logerr("Failed to switch to OFFBOARD after 5 attempts")
                    self.controller_started = False
                    return
                
                if self.set_mode_client.call(base_mode=0, custom_mode="OFFBOARD").mode_sent:
                    rospy.loginfo("OFFBOARD enabled")
                else:
                    rospy.logwarn("OFFBOARD switch failed, retrying...")
                    retry_count += 1
                rospy.sleep(1)
            
            # 4. 所有条件满足，启动MPC
            if self.current_state.armed and self.current_state.mode == "OFFBOARD":
                self.controller_start_time = rospy.Time.now()
                
            else:
                rospy.logerr("Failed to meet all conditions for MPC start")
                self.controller_started = False
                
        except Exception as e:
            rospy.logerr(f"Error in prepare_flight: {str(e)}")
            self.controller_started = False

    def callback_joint_state(self, msg):
        """Handle joint state messages for the robotic arm"""
        if not self.node_params.arm_enable:
            return

        # Check if the joint name starts with arm_name
        if not msg.name[0].startswith(self.node_params.arm_name):
            return
        
        with self.state_lock:
            # Copy current state to avoid race conditions
            state_new = np.copy(self.state)
            
            # Update joint positions and velocities
            for i in range(len(msg.position)):
                # Joint positions start after floating base (7 values)
                state_new[7 + i] = msg.position[i]
                # Joint velocities start after floating base velocities (nq + 6 values)
                state_new[self.mpc_controller.robot_model.nq + 6 + i] = msg.velocity[i]
            
            # Update the state
            self.state = state_new

        # Record solver performance if enabled
        if self.node_params.record_solver and self.node_params.record_solver_level > 0:
            # Note: This part needs to be implemented if you need solver performance recording
            pass

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
        state_msg.position.z = -self.state[2]
        state_msg.orientation.x = self.state[3]
        state_msg.orientation.y = self.state[4]
        state_msg.orientation.z = self.state[5]
        state_msg.orientation.w = self.state[6]
        
        state_msg.mpc_time_step = self.mpc_time_step
        state_msg.solving_time = self.solving_time
        
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
                
            # 打印位置和姿态（用于调试）
            # rospy.loginfo(f"Iris position: {pose.position}")
            # rospy.loginfo(f"Iris orientation: {pose.orientation}")
            rospy.logdebug(f"Iris state: {self.state}")
            
        except ValueError:
            rospy.logwarn("Could not find iris model in gazebo model states")

    def callback_state(self, msg):
        """处理MAVROS状态回调"""
        self.current_state = msg

if __name__ == '__main__':
    print(crocoddyl.__version__)
    print(crocoddyl.__file__)
    rospy.init_node('mpc_runner')
    try:
        mpc_runner = MpcRunner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 