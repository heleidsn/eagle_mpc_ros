///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, Institut de Robotica i Informatica Industrial (CSIC-UPC)
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////


#include "mpc_runner.hpp"

#include "eagle_mpc/utils/tools.hpp"

MpcRunner::MpcRunner() {
  initializeParameters();
  initializeMpcController();
  initializeVariables();
  initializeMsgs();
  initializeSubscribers();

  controller_started_ = false;
  initializePublishers();
  // RQT Config
  server_ = boost::make_shared<dynamic_reconfigure::Server<eagle_mpc_controller::ParamsConfig>>(server_mutex_);
  callback_server_ = boost::bind(&MpcRunner::callbackConfig, this, _1, _2);
  server_->setCallback(callback_server_);
  node_params_.initialized_time = ros::Time::now();
}

MpcRunner::~MpcRunner() {}

void MpcRunner::initializeParameters() {
  // Trajectory related
  nh_.param<std::string>(ros::this_node::getNamespace() + "/trajectory_config", node_params_.trajectory_config_path,
                         "");
  int trajectory_dt;
  nh_.param<int>(ros::this_node::getNamespace() + "/trajectory_dt", trajectory_dt, 10);
  node_params_.trajectory_dt = (std::size_t)trajectory_dt;

  std::string trajectory_solver;
  nh_.param<std::string>(ros::this_node::getNamespace() + "/trajectory_solver", trajectory_solver, "SolverSbFDDP");
  node_params_.trajectory_solver = eagle_mpc::SolverTypes_map.at(trajectory_solver);

  nh_.param<std::string>(ros::this_node::getNamespace() + "/trajectory_integration",
                         node_params_.trajectory_integration, "IntegratedActionModelEuler");
  node_params_.trajectory_squash = node_params_.trajectory_solver == eagle_mpc::SolverTypes::SolverSbFDDP;

  nh_.param<std::string>(ros::this_node::getNamespace() + "/mpc_config", node_params_.mpc_config_path, "");
  nh_.param<std::string>(ros::this_node::getNamespace() + "/mpc_type", node_params_.mpc_type, "");
  nh_.param<bool>(ros::this_node::getNamespace() + "/use_internal_gains", node_params_.use_internal_gains, false);
  nh_.param<bool>(ros::this_node::getNamespace() + "/record_solver", node_params_.record_solver, false);
  int solver_level;
  nh_.param<int>(ros::this_node::getNamespace() + "/record_solver_level", solver_level, 0);
  node_params_.record_solver_level = (std::size_t)solver_level;
  nh_.param<bool>(ros::this_node::getNamespace() + "/automatic_start", node_params_.automatic_start, false);
  nh_.param<std::string>(ros::this_node::getNamespace() + "/arm_name", node_params_.arm_name, "");
  node_params_.arm_enable = node_params_.arm_name != "";

  int motor_command_dt;
  nh_.param<int>(ros::this_node::getNamespace() + "/motor_command_dt", motor_command_dt, 0);
  node_params_.motor_command_dt = (std::size_t)motor_command_dt;
}

void MpcRunner::initializeMpcController() {
  trajectory_ = eagle_mpc::Trajectory::create();
  trajectory_->autoSetup(node_params_.trajectory_config_path);

  ROS_WARN("This is the trajectory dt: %ld", node_params_.trajectory_dt);
  boost::shared_ptr<crocoddyl::ShootingProblem> problem = trajectory_->createProblem(
      node_params_.trajectory_dt, node_params_.trajectory_squash, node_params_.trajectory_integration);

  boost::shared_ptr<crocoddyl::SolverFDDP> solver;
  switch (node_params_.trajectory_solver) {
    case eagle_mpc::SolverTypes::SolverSbFDDP:
      solver = boost::make_shared<eagle_mpc::SolverSbFDDP>(problem, trajectory_->get_squash());
      break;
    case eagle_mpc::SolverTypes::SolverBoxFDDP:
      solver = boost::make_shared<crocoddyl::SolverBoxFDDP>(problem);
      break;
  }
  std::vector<boost::shared_ptr<crocoddyl::CallbackAbstract>> callbacks;
  callbacks.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());
  solver->setCallbacks(callbacks);
  solver->solve(crocoddyl::DEFAULT_VECTOR, crocoddyl::DEFAULT_VECTOR);

  switch (eagle_mpc::MpcTypes_map.at(node_params_.mpc_type)) {
    case eagle_mpc::MpcTypes::Carrot:
      mpc_controller_ = boost::make_shared<eagle_mpc::CarrotMpc>(
          trajectory_, solver->get_xs(), node_params_.trajectory_dt, node_params_.mpc_config_path);
      break;
    case eagle_mpc::MpcTypes::Rail:
      mpc_controller_ = boost::make_shared<eagle_mpc::RailMpc>(solver->get_xs(), node_params_.trajectory_dt,
                                                                     node_params_.mpc_config_path);
      break;
    case eagle_mpc::MpcTypes::Weighted:
      mpc_controller_ = boost::make_shared<eagle_mpc::WeightedMpc>(trajectory_, node_params_.trajectory_dt,
                                                                         node_params_.mpc_config_path);
  }

  if (mpc_controller_->get_solver_type() == eagle_mpc::SolverTypes::SolverSbFDDP) {
    boost::static_pointer_cast<eagle_mpc::SolverSbFDDP>(mpc_controller_->get_solver())
        ->set_convergence_init(1e-3);
  }
}

void MpcRunner::initializeVariables() {
  if (node_params_.use_internal_gains) {
    motor_command_dt_ = node_params_.motor_command_dt / 1000.0;
  } else {
    motor_command_dt_ = mpc_controller_->get_dt() / 1000.0;
  }

  // States
  state_ = mpc_controller_->get_robot_state()->zero();
  state_ref_ = state_;
  state_diff_ = Eigen::VectorXd::Zero(mpc_controller_->get_robot_state()->get_ndx());

  // Thrust command
  control_command_ = Eigen::VectorXd::Zero(mpc_controller_->get_actuation()->get_nu());
  thrust_command_ = Eigen::VectorXd::Zero(mpc_controller_->get_platform_params()->n_rotors_);
  speed_command_ = thrust_command_;
  if (control_command_.size() - thrust_command_.size() > 0) {
    torque_command_ = Eigen::VectorXd::Zero(control_command_.size() - thrust_command_.size());
  }

  // Gains
  flag_new_gains_ = false;
  fb_gains_ =
      Eigen::MatrixXd::Zero(mpc_controller_->get_actuation()->get_nu(), mpc_controller_->get_robot_state()->get_ndx());

  node_params_.start_seconds = 10;
}

void MpcRunner::initializeMsgs() {
  msg_thrusts_.angular_velocities.resize(thrust_command_.size());
  msg_motors_state_.thrusts.resize(thrust_command_.size());
  if (node_params_.record_solver and node_params_.record_solver_level > 0) {
    if (control_command_.size() - thrust_command_.size() > 0) {
      msg_solver_performance_.joints_initial.resize(control_command_.size() - thrust_command_.size());
    }
    msg_solver_performance_.floating_base_trajectory.resize(mpc_controller_->get_knots());
  }
}

void MpcRunner::initializeSubscribers() {
  // Subscribers
  if (node_params_.use_internal_gains) {
    subs_odom_ =
        nh_.subscribe("/odometry", 1, &MpcRunner::callbackOdometryGains, this, ros::TransportHints().tcpNoDelay());
    timer_mpc_solve_ =
        nh_.createTimer(ros::Duration(mpc_controller_->get_dt() / 1000.0), &MpcRunner::callbackMpcSolve, this);
    ROS_INFO_STREAM("Publishing motor control every: " << motor_command_dt_ << " seconds");
  } else {
    subs_odom_ =
        nh_.subscribe("/odometry", 1, &MpcRunner::callbackOdometryMpc, this, ros::TransportHints().tcpNoDelay());
  }

  if (node_params_.arm_enable) {
    subs_joint_state_ =
        nh_.subscribe("/joint_states", 1, &MpcRunner::callbackJointState, this, ros::TransportHints().tcpNoDelay());
  }

  if (node_params_.automatic_start) {
    timer_auto_start_ = nh_.createTimer(ros::Duration(1), &MpcRunner::callbackCountdown, this);
  }
}

void MpcRunner::initializePublishers() {
  pub_thrust_command_ = nh_.advertise<mav_msgs::Actuators>("/motor_command", 1);
  if (node_params_.record_solver) {
    pub_solver_performance_ = nh_.advertise<eagle_mpc_msgs::SolverPerformance>("/solver_performance", 1);
  }

  if (node_params_.arm_enable) {
    std::size_t n_joints = mpc_controller_->get_robot_model()->nq - 7;
    pub_arm_commands_.reserve(n_joints);
    for (std::size_t i = 0; i < n_joints; ++i) {
      pub_arm_commands_.push_back(nh_.advertise<std_msgs::Float64>("/joint_command_" + std::to_string(i + 1), 1));
    }
  }
}

void MpcRunner::callbackOdometryMpc(const nav_msgs::OdometryConstPtr &msg_odometry) {
  mut_state_.lock();
  state_(0) = msg_odometry->pose.pose.position.x;
  state_(1) = msg_odometry->pose.pose.position.y;
  state_(2) = msg_odometry->pose.pose.position.z;
  state_(3) = msg_odometry->pose.pose.orientation.x;
  state_(4) = msg_odometry->pose.pose.orientation.y;
  state_(5) = msg_odometry->pose.pose.orientation.z;
  state_(6) = msg_odometry->pose.pose.orientation.w;
  state_(mpc_controller_->get_robot_model()->nq) = msg_odometry->twist.twist.linear.x;
  state_(mpc_controller_->get_robot_model()->nq + 1) = msg_odometry->twist.twist.linear.y;
  state_(mpc_controller_->get_robot_model()->nq + 2) = msg_odometry->twist.twist.linear.z;
  state_(mpc_controller_->get_robot_model()->nq + 3) = msg_odometry->twist.twist.angular.x;
  state_(mpc_controller_->get_robot_model()->nq + 4) = msg_odometry->twist.twist.angular.y;
  state_(mpc_controller_->get_robot_model()->nq + 5) = msg_odometry->twist.twist.angular.z;
  state_time_ = msg_odometry->header.stamp;
  mut_state_.unlock();

  if (controller_started_) {
    mut_state_.lock();
    mpc_controller_->get_problem()->set_x0(state_);
    mut_state_.unlock();

    controller_time_ = ros::Time::now() - controller_start_time_;
    controller_instant_ = (std::size_t)(controller_time_.toSec() * 1000.0);
    mpc_controller_->updateProblem(controller_instant_);

    if (node_params_.record_solver) {
      solver_time_init_ = ros::WallTime::now();
    }
    mpc_controller_->get_solver()->solve(mpc_controller_->get_solver()->get_xs(),
                                         mpc_controller_->get_solver()->get_us(), mpc_controller_->get_iters());

    // PROPERLY HANDLE SOLVER TYPE!
    control_command_ = boost::static_pointer_cast<eagle_mpc::SolverSbFDDP>(mpc_controller_->get_solver())
                           ->getSquashControls()[0];

    thrust_command_ = control_command_.head(thrust_command_.size());
    eagle_mpc::Tools::thrustToSpeed(thrust_command_, mpc_controller_->get_platform_params(), speed_command_);
    if (ros::Duration(ros::Time::now() - control_last_).toSec() > 1.5 * mpc_controller_->get_dt()) {
      ROS_WARN("Control rate not accomplished!");
    }
    control_last_ = ros::Time::now();

    msg_thrusts_.header.stamp = ros::Time::now();
    for (std::size_t i = 0; i < speed_command_.size(); ++i) {
      msg_thrusts_.angular_velocities[i] = speed_command_(i);
    }
    pub_thrust_command_.publish(msg_thrusts_);

    for (std::size_t i = 0; i < mpc_controller_->get_robot_model()->nq - 7; ++i) {
      msg_joint_command_.data = control_command_(speed_command_.size() + i);
      pub_arm_commands_[i].publish(msg_joint_command_);
    }

    if (node_params_.record_solver) {
      publishSolver(msg_odometry);
    }
  }
}

void MpcRunner::callbackOdometryGains(const nav_msgs::OdometryConstPtr &msg_odometry) {
  // mut_state_.lock();
  // state_(0) = msg_odometry->pose.pose.position.x;
  // state_(1) = msg_odometry->pose.pose.position.y;
  // state_(2) = msg_odometry->pose.pose.position.z;
  // state_(3) = msg_odometry->pose.pose.orientation.x;
  // state_(4) = msg_odometry->pose.pose.orientation.y;
  // state_(5) = msg_odometry->pose.pose.orientation.z;
  // state_(6) = msg_odometry->pose.pose.orientation.w;
  // state_(7) = msg_odometry->twist.twist.linear.x;
  // state_(8) = msg_odometry->twist.twist.linear.y;
  // state_(9) = msg_odometry->twist.twist.linear.z;
  // state_(10) = msg_odometry->twist.twist.angular.x;
  // state_(11) = msg_odometry->twist.twist.angular.y;
  // state_(12) = msg_odometry->twist.twist.angular.z;
  // state_time_ = msg_odometry->header.stamp;
  // mut_state_.unlock();

  // if (controller_started_) {
  //   if (flag_new_gains_) {
  //     mut_gains_.lock();
  //     motors_thrust_ = motors_thrust_new_;
  //     fb_gains_ = fb_gains_new_;
  //     flag_new_gains_ = false;
  //     mut_gains_.unlock();
  //   }

  //   mut_state_.lock();
  //   ros::Duration elapsed_time = ros::Time::now() - control_last_;
  //   if (elapsed_time > ros::Duration(motor_command_dt_)) {
  //     ROS_WARN("Control rate not accomplished!");
  //   }
  //   mpc_main_.getMpcController()->getStateMultibody()->diff(state_ref_, state_, state_diff_);
  //   mut_state_.unlock();

  //   mut_control_.lock();
  //   motors_thrust_ = motors_thrust_ - fb_gains_ * state_diff_;
  //   motors_thrust_ = (motors_thrust_.array() < 0).select(0, motors_thrust_);  // Avoid negative numbers
  //   mpc_main_.thrustToSpeed(motors_thrust_, motors_speed_);
  //   mut_control_.unlock();

  //   msg_actuators_.header.stamp = ros::Time::now();
  //   msg_actuators_.angular_velocities[0] = motors_speed_(0);
  //   msg_actuators_.angular_velocities[1] = motors_speed_(1);
  //   msg_actuators_.angular_velocities[2] = motors_speed_(2);
  //   msg_actuators_.angular_velocities[3] = motors_speed_(3);
  //   pub_motor_command_.publish(msg_actuators_);
  // }
}

void MpcRunner::callbackJointState(const sensor_msgs::JointStateConstPtr &msg_joint_state) {
  if (msg_joint_state->name[0].compare(0, node_params_.arm_name.size(), node_params_.arm_name) == 0) {
    mut_state_.lock();
    for (std::size_t i = 0; i < msg_joint_state->position.size(); ++i) {
      state_(7 + i) = msg_joint_state->position[i];
      state_(mpc_controller_->get_robot_model()->nq + 6 + i) = msg_joint_state->velocity[i];
    }
    mut_state_.unlock();
    if (node_params_.record_solver && node_params_.record_solver_level > 0) {
      for (std::size_t i = 0; i < msg_solver_performance_.joints_initial.size(); ++i) {
        msg_solver_performance_.joints_initial[i].position = msg_joint_state->position[i];
        msg_solver_performance_.joints_initial[i].velocity = msg_joint_state->velocity[i];
      }
    }
  }
}

void MpcRunner::callbackMpcSolve(const ros::TimerEvent &) {
  // if (controller_started_) {
  //   mut_state_.lock();
  //   ros::Duration elapsed_time = ros::Time::now() - state_time_;
  //   if (elapsed_time > ros::Duration(0.01)) {
  //     ROS_WARN("MPC Solve: Outdated state");
  //   }
  //   // Here we should account for the solving delay
  //   mpc_main_.setCurrentState(state_);
  //   state_ref_ = state_;
  //   mut_state_.unlock();

  //   mpc_main_.runMpcStep();

  //   mut_gains_.lock();
  //   control_last_ = ros::Time::now();
  //   motors_thrust_new_ = mpc_main_.getMotorsThrust();
  //   fb_gains_new_ = mpc_main_.getFeedBackGains();
  //   flag_new_gains_ = true;
  //   mut_gains_.unlock();
  // }
}

void MpcRunner::callbackCountdown(const ros::TimerEvent &) {
  node_params_.start_seconds -= 1;
  ROS_WARN("Missinon Countdown: %ld", node_params_.start_seconds);
  if (node_params_.start_seconds == 0) {
    eagle_mpc_controller::ParamsConfig params;
    params.start_mission = true;
    boost::recursive_mutex::scoped_lock lock(server_mutex_);
    server_->updateConfig(params);
    lock.unlock();
    timer_auto_start_.stop();
    controller_started_ = true;
    controller_start_time_ = ros::Time::now();
  }
}

void MpcRunner::callbackConfig(eagle_mpc_controller::ParamsConfig &config, uint32_t level) {
  ROS_INFO("RECONFIGURE_REQUEST!");
  controller_started_ = config.start_mission;
  if (controller_started_) {
    controller_start_time_ = ros::Time::now();
  }
}

void MpcRunner::publishSolver(const nav_msgs::OdometryConstPtr &msg_odometry) {
  solver_duration_ = ros::WallTime::now() - solver_time_init_;
  msg_solver_performance_.header.stamp = ros::Time::now();
  msg_solver_performance_.solving_time.sec = solver_duration_.sec;
  msg_solver_performance_.solving_time.nsec = solver_duration_.nsec;
  msg_solver_performance_.iters = mpc_controller_->get_solver()->get_iter();
  msg_solver_performance_.final_cost = mpc_controller_->get_solver()->get_cost();
  msg_solver_performance_.initial_time = controller_time_;

  if (node_params_.record_solver_level > 0) {
    msg_solver_performance_.floating_base_initial.pose = msg_odometry->pose.pose;
    msg_solver_performance_.floating_base_initial.motion = msg_odometry->twist.twist;

    for (std::size_t i = 0; i < mpc_controller_->get_knots(); ++i) {
      msg_solver_performance_.floating_base_trajectory[i].pose.position.x =
          mpc_controller_->get_solver()->get_xs()[i](0);
      msg_solver_performance_.floating_base_trajectory[i].pose.position.y =
          mpc_controller_->get_solver()->get_xs()[i](1);
      msg_solver_performance_.floating_base_trajectory[i].pose.position.z =
          mpc_controller_->get_solver()->get_xs()[i](2);
    }
  }

  pub_solver_performance_.publish(msg_solver_performance_);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, ros::this_node::getName());

  MpcRunner mpc_runner;

  ros::AsyncSpinner s(0);
  s.start();
  ros::waitForShutdown();
}