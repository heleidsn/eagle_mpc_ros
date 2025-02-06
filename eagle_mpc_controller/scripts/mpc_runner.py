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
        self.initialize_parameters()
        self.initialize_mpc_controller()
        self.initialize_variables()
        self.initialize_msgs()
        self.initialize_subscribers()

        rospy.loginfo("MpcRunner init ok")

        self.controller_started = False
        self.initialize_publishers()
        
        # RQT Config
        self.server = Server(ParamsConfig, self.callback_config)
        self.node_params.initialized_time = rospy.Time.now()

    def initialize_parameters(self):
        # Create a namespace for parameters
        self.node_params = type('NodeParams', (), {})()
        
        # Trajectory related
        self.node_params.trajectory_config_path = rospy.get_param('~trajectory_config', '')
        trajectory_dt = rospy.get_param('~trajectory_dt', 10)
        self.node_params.trajectory_dt = trajectory_dt

        trajectory_solver = rospy.get_param('~trajectory_solver', 'SolverSbFDDP')
        self.node_params.trajectory_solver = eagle_mpc.SolverTypes_map[trajectory_solver]

        self.node_params.trajectory_integration = rospy.get_param('~trajectory_integration', 'IntegratedActionModelEuler')
        self.node_params.trajectory_squash = self.node_params.trajectory_solver == eagle_mpc.SolverTypes.SolverSbFDDP

        self.node_params.mpc_config_path = rospy.get_param('~mpc_config', '')
        self.node_params.mpc_type = rospy.get_param('~mpc_type', '')
        self.node_params.use_internal_gains = rospy.get_param('~use_internal_gains', False)
        self.node_params.record_solver = rospy.get_param('~record_solver', False)
        self.node_params.record_solver_level = rospy.get_param('~record_solver_level', 0)
        self.node_params.automatic_start = rospy.get_param('~automatic_start', False)
        self.node_params.arm_name = rospy.get_param('~arm_name', '')
        self.node_params.arm_enable = bool(self.node_params.arm_name)

        self.node_params.motor_command_dt = rospy.get_param('~motor_command_dt', 0)

        rospy.loginfo("initializeParameters ok")

    def initialize_mpc_controller(self):
        rospy.logwarn("initializeMpcController start")
        self.trajectory = eagle_mpc.Trajectory.create()
        self.trajectory.autoSetup(self.node_params.trajectory_config_path)
        rospy.logwarn("initializeMpcController ok")

        problem = self.trajectory.createProblem(
            self.node_params.trajectory_dt,
            self.node_params.trajectory_squash,
            self.node_params.trajectory_integration
        )

        if self.node_params.trajectory_solver == eagle_mpc.SolverTypes.SolverSbFDDP:
            solver = eagle_mpc.SolverSbFDDP(problem, self.trajectory.get_squash())
        else:
            solver = crocoddyl.SolverBoxFDDP(problem)

        callbacks = [crocoddyl.CallbackVerbose()]
        solver.setCallbacks(callbacks)
        solver.solve()

        # Initialize MPC controller based on type
        mpc_type = eagle_mpc.MpcTypes_map[self.node_params.mpc_type]
        if mpc_type == eagle_mpc.MpcTypes.Carrot:
            self.mpc_controller = eagle_mpc.CarrotMpc(
                self.trajectory,
                solver.get_xs(),
                self.node_params.trajectory_dt,
                self.node_params.mpc_config_path
            )
        elif mpc_type == eagle_mpc.MpcTypes.Rail:
            self.mpc_controller = eagle_mpc.RailMpc(
                solver.get_xs(),
                self.node_params.trajectory_dt,
                self.node_params.mpc_config_path
            )
        elif mpc_type == eagle_mpc.MpcTypes.Weighted:
            self.mpc_controller = eagle_mpc.WeightedMpc(
                self.trajectory,
                self.node_params.trajectory_dt,
                self.node_params.mpc_config_path
            )

        rospy.loginfo("initializeMpcController ok")

    def initialize_variables(self):
        # Initialize state variables
        self.state_lock = Lock()
        self.state = self.mpc_controller.get_robot_state().zero()
        self.state_ref = self.state.copy()
        self.state_diff = np.zeros(self.mpc_controller.get_robot_state().get_ndx())

        # Initialize control variables
        self.control_command = np.zeros(self.mpc_controller.get_actuation().get_nu())
        self.thrust_command = np.zeros(self.mpc_controller.get_platform_params().n_rotors_)
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
        if self.node_params.use_internal_gains:
            self.odom_sub = rospy.Subscriber(
                "/odometry", Odometry, self.callback_odometry_gains, tcp_nodelay=True)
            self.mpc_timer = rospy.Timer(
                rospy.Duration(self.mpc_controller.get_dt() / 1000.0),
                self.callback_mpc_solve
            )
        else:
            self.odom_sub = rospy.Subscriber(
                "/odometry", Odometry, self.callback_odometry_mpc, tcp_nodelay=True)

        if self.node_params.arm_enable:
            self.joint_state_sub = rospy.Subscriber(
                "/joint_states", JointState, self.callback_joint_state, tcp_nodelay=True)

        if self.node_params.automatic_start:
            self.auto_start_timer = rospy.Timer(rospy.Duration(1), self.callback_countdown)

    def initialize_publishers(self):
        self.thrust_pub = rospy.Publisher("/motor_command", Actuators, queue_size=1)
        
        if self.node_params.record_solver:
            self.solver_performance_pub = rospy.Publisher(
                "/solver_performance", SolverPerformance, queue_size=1)

        if self.node_params.arm_enable:
            n_joints = self.mpc_controller.get_robot_model().nq - 7
            self.arm_pubs = []
            for i in range(n_joints):
                self.arm_pubs.append(
                    rospy.Publisher(f"/joint_command_{i+1}", Float64, queue_size=1))

    def callback_odometry_mpc(self, msg):
        with self.state_lock:
            # Update state from odometry message
            self.state[0:3] = [msg.pose.pose.position.x, 
                             msg.pose.pose.position.y,
                             msg.pose.pose.position.z]
            self.state[3:7] = [msg.pose.pose.orientation.x,
                              msg.pose.pose.orientation.y,
                              msg.pose.pose.orientation.z,
                              msg.pose.pose.orientation.w]
            nq = self.mpc_controller.get_robot_model().nq
            self.state[nq:nq+3] = [msg.twist.twist.linear.x,
                                  msg.twist.twist.linear.y,
                                  msg.twist.twist.linear.z]
            self.state[nq+3:nq+6] = [msg.twist.twist.angular.x,
                                    msg.twist.twist.angular.y,
                                    msg.twist.twist.angular.z]

        if self.controller_started:
            self.run_mpc_iteration(msg)

    def run_mpc_iteration(self, msg):
        # Set initial state and update problem
        with self.state_lock:
            self.mpc_controller.get_problem().set_x0(self.state)

        self.controller_time = rospy.Time.now() - self.controller_start_time
        controller_instant = int(self.controller_time.to_sec() * 1000.0)
        self.mpc_controller.updateProblem(controller_instant)

        # Solve MPC problem
        if self.node_params.record_solver:
            self.solver_time_init = time.time()
            
        self.mpc_controller.get_solver().solve(
            self.mpc_controller.get_solver().get_xs(),
            self.mpc_controller.get_solver().get_us(),
            self.mpc_controller.get_iters()
        )

        # Get control commands and publish
        self.publish_commands()
        
        if self.node_params.record_solver:
            self.publish_solver_performance(msg)

    def publish_commands(self):
        # Get control commands
        self.control_command = self.mpc_controller.get_solver().getSquashControls()[0]
        self.thrust_command = self.control_command[:len(self.thrust_command)]
        
        # Convert thrust to speed
        eagle_mpc.Tools.thrustToSpeed(
            self.thrust_command,
            self.mpc_controller.get_platform_params(),
            self.speed_command
        )

        # Publish thrust commands
        self.msg_thrusts.header.stamp = rospy.Time.now()
        self.msg_thrusts.angular_velocities = self.speed_command.tolist()
        self.thrust_pub.publish(self.msg_thrusts)

        # Publish joint commands if arm is enabled
        if self.node_params.arm_enable:
            for i, pub in enumerate(self.arm_pubs):
                msg = Float64()
                msg.data = self.control_command[len(self.speed_command) + i]
                pub.publish(msg)

    def callback_config(self, config, level):
        rospy.loginfo("RECONFIGURE_REQUEST!")
        self.controller_started = config.start_mission
        if self.controller_started:
            self.controller_start_time = rospy.Time.now()

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