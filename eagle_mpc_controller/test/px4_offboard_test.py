#!/usr/bin/env python
import rospy
from geometry_msgs.msg import TwistStamped
from mavros_msgs.srv import SetMode, CommandBool

def set_mode(mode):
    rospy.wait_for_service('/mavros/set_mode')
    try:
        set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        response = set_mode_client(custom_mode=mode)
        return response.mode_sent
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to set mode: {e}")
        return False

def arm():
    rospy.wait_for_service('/mavros/cmd/arming')
    try:
        arm_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        response = arm_client(value=True)
        return response.success
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to arm: {e}")
        return False

def send_angular_velocity():
    rospy.init_node('px4_angular_velocity_control', anonymous=True)
    vel_pub = rospy.Publisher('/mavros/setpoint_attitude/cmd_vel', TwistStamped, queue_size=10)
    rate = rospy.Rate(20)  # 20 Hz

    # 进入 OFFBOARD 模式
    rospy.sleep(2)  # 等待连接
    if set_mode("OFFBOARD"):
        rospy.loginfo("Offboard mode set")
    if arm():
        rospy.loginfo("Drone armed")

    # 发送角速度命令
    while not rospy.is_shutdown():
        cmd = TwistStamped()
        cmd.twist.angular.x = 0.0  # 无滚转角速度
        cmd.twist.angular.y = 0.0  # 无俯仰角速度
        cmd.twist.angular.z = 0.5  # 设定偏航角速度 0.5 rad/s
        vel_pub.publish(cmd)
        rate.sleep()

if __name__ == '__main__':
    try:
        send_angular_velocity()
    except rospy.ROSInterruptException:
        pass
