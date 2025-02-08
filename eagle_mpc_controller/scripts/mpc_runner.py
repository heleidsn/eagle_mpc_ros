#!/usr/bin/env python

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from mav_msgs.msg import Actuators
from std_msgs.msg import Float64
from eagle_mpc_msgs.msg import SolverPerformance
from dynamic_reconfigure.server import Server
from eagle_mpc_controller.cfg import ParamsConfig
import eagle_mpc
import crocoddyl
from threading import Lock
import time

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

        self.initialize_parameters()
        self.initialize_mpc_controller()
        self.initialize_variables()
        self.initialize_msgs()
        self.initialize_subscribers()

        rospy.loginfo("MpcRunner init ok")

        self.controller_started = False
        self.initialize_publishers()
        
        try:
            # RQT Config
            self.server = Server(ParamsConfig, self.callback_config)
            self.node_params.initialized_time = rospy.Time.now()
        except Exception as e:
            rospy.logerr("Failed to initialize dynamic reconfigure server: %s", str(e))
            raise

    def initialize_parameters(self):
        # Create a namespace for parameters
        self.node_params = type('NodeParams', (), {})()
        
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
        except Exception as e:
            rospy.logerr("Failed to create MPC controller: %s", str(e))
            raise

        rospy.loginfo("initializeMpcController ok")

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
        
        if self.node_params.record_solver:
            self.msg_solver_performance = SolverPerformance()
            if self.node_params.record_solver_level > 0:
                self.msg_solver_performance.floating_base_trajectory = [None] * self.mpc_controller.get_knots()

    def initialize_subscribers(self):
        
        self.odom_sub = rospy.Subscriber(
            "/hexacopter370/ground_truth/odometry", Odometry, self.callback_odometry_mpc, tcp_nodelay=True)

        if self.node_params.arm_enable:
            self.joint_state_sub = rospy.Subscriber(
                "/hexacopter370/joint_states", JointState, self.callback_joint_state, tcp_nodelay=True)

        if self.node_params.automatic_start:
            self.auto_start_timer = rospy.Timer(rospy.Duration(1), self.callback_countdown)

    def initialize_publishers(self):
        self.thrust_pub = rospy.Publisher("/hexacopter370/command/motor_speed", Actuators, queue_size=1)
        
        if self.node_params.record_solver:
            self.solver_performance_pub = rospy.Publisher(
                "/solver_performance", SolverPerformance, queue_size=1)

        if self.node_params.arm_enable:
            n_joints = self.mpc_controller.robot_model.nq - 7
            self.arm_pubs = []
            for i in range(n_joints):
                self.arm_pubs.append(
                    rospy.Publisher(f"/joint_command_{i+1}", Float64, queue_size=1))

    def callback_odometry_mpc(self, msg):
        with self.state_lock:
            # Create temporary array and assign values
            state_new = np.copy(self.state)
            state_new[0:3] = [msg.pose.pose.position.x,
                           msg.pose.pose.position.y,
                           msg.pose.pose.position.z]
            state_new[3:7] = [msg.pose.pose.orientation.x,
                           msg.pose.pose.orientation.y,
                           msg.pose.pose.orientation.z,
                           msg.pose.pose.orientation.w]
            nq = self.mpc_controller.state.nq  # 使用state属性获取nq
            state_new[nq:nq+3] = [msg.twist.twist.linear.x,
                               msg.twist.twist.linear.y,
                               msg.twist.twist.linear.z]
            state_new[nq+3:nq+6] = [msg.twist.twist.angular.x,
                                 msg.twist.twist.angular.y,
                                 msg.twist.twist.angular.z]
            # 将新状态赋值给self.state
            self.state = state_new

        if self.controller_started:
            self.run_mpc_iteration(msg)

    def run_mpc_iteration(self, msg):
        # Set initial state and update problem
        with self.state_lock:
            self.mpc_controller.problem.x0 = self.state

        self.controller_time = rospy.Time.now() - self.controller_start_time
        controller_instant = int(self.controller_time.to_sec() * 1000.0)
        self.mpc_controller.updateProblem(controller_instant)

        # Solve MPC problem
        if self.node_params.record_solver:
            self.solver_time_init = time.time()
            
        self.mpc_controller.solver.solve(
            self.mpc_controller.solver.xs,
            self.mpc_controller.solver.us,
            self.mpc_controller.iters
        )

        # Get control commands and publish
        self.publish_commands()
        
        if self.node_params.record_solver:
            self.publish_solver_performance(msg)

    def publish_commands(self):
        # Get control commands
        self.control_command = self.mpc_controller.solver.us_squash[0]
        self.thrust_command = self.control_command[:len(self.thrust_command)]
        
        self.speed_command = self.thrust_command / self.mpc_controller.platform_params.cf

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

    def callback_config(self, config, level):
        rospy.loginfo("RECONFIGURE_REQUEST!")
        self.controller_started = config.start_mission
        if self.controller_started:
            self.controller_start_time = rospy.Time.now()
        return config  # 必须返回配置对象

    def callback_countdown(self, event):
        self.node_params.start_seconds -= 1
        rospy.logwarn(f"Mission Countdown: {self.node_params.start_seconds}")
        if self.node_params.start_seconds == 0:
            params = ParamsConfig.defaults
            params.start_mission = True
            self.server.update_configuration(params)
            self.controller_started = True
            self.controller_start_time = rospy.Time.now()

if __name__ == '__main__':
    rospy.init_node('mpc_runner')
    try:
        mpc_runner = MpcRunner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 