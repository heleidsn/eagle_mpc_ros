#!/usr/bin/env python
'''
Author: Lei He
Date: 2025-02-19 17:29:01
LastEditTime: 2025-02-20 08:41:45
Description: New version of MPC runner for PX4 SITL with eagle MPC
Github: https://github.com/heleidsn
'''
import rospy
from mpc_debug.utils.create_problem import get_opt_traj, create_mpc_controller
from eagle_mpc_msgs.msg import SolverPerformance, MpcState, MpcControl
import numpy as np
from nav_msgs.msg import Odometry


class MpcRunner:
    def __init__(self):
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
        
        # start MPC controller and wait for it to start
        self.mpc_rate = 20.0  # Hz
        self.mpc_timer = rospy.Timer(rospy.Duration(1.0/self.mpc_rate), self.mpc_timer_callback)
        
        self.odom_sub = rospy.Subscriber("/gazebo/model_states", Odometry, self.callback_model_state_gazebo, tcp_nodelay=True)
        
    
    def mpc_timer_callback(self, event):
        # Set initial state and update problem
        self.mpc_controller.problem.x0 = self.state
            
        if self.controller_started:
            self.controller_time = rospy.Time.now() - self.controller_start_time
            self.mpc_ref_index = int(self.controller_time.to_sec() * 1000.0 / self.node_params.trajectory_dt)
        else:
            self.mpc_ref_index = 0
        
        # update problem
        self.mpc_controller.updateProblem(self.mpc_ref_index)
        
        time_start = rospy.Time.now()
        self.mpc_controller.solver.solve(
            self.mpc_controller.solver.xs,
            self.mpc_controller.solver.us,
            self.mpc_controller.iters
        )
        time_end = rospy.Time.now()
        self.solving_time = (time_end - time_start).to_sec()
        
        rospy.loginfo("mpc_controller.solver.xs: %s", self.mpc_controller.solver.xs[0])
        rospy.loginfo("mpc_controller.solver.us: %s", self.mpc_controller.solver.us_squash[0])

        # publish control command
        self.publish_mavros_rate_command()
        
        # publish mpc debug data
        self.publish_mpc_data()
        
        
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





if __name__ == '__main__':
    rospy.init_node('mpc_runner')
    try:
        mpc_runner = MpcRunner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass