import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State, AttitudeTarget, ActuatorControl
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest

from geometry_msgs.msg import Vector3
import time

current_state = State()

def state_cb(msg):
    global current_state
    current_state = msg
    
def send_body_rate_thrust():
    rospy.init_node('body_rate_thrust_control', anonymous=True)
    attitude_pub = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)

    rate = rospy.Rate(50)  # 50Hz 控制频率

    # 等待 MAVROS 连接
    time.sleep(2)

    while not rospy.is_shutdown():
        att_msg = AttitudeTarget()
        att_msg.header.stamp = rospy.Time.now()
        
        # 设置 type_mask，忽略姿态，仅使用角速度 + 推力
        att_msg.type_mask = AttitudeTarget.IGNORE_ATTITUDE 
        
        # 机体系角速度 (rad/s)
        att_msg.body_rate = Vector3(0.0, 0.0, 0.1)  # 仅绕 Z 轴旋转 0.1 rad/s
        
        # 推力值 (范围 0 ~ 1)
        att_msg.thrust = 0.7  # 60% 油门

        attitude_pub.publish(att_msg)
        rate.sleep()

def send_motor_commands():
    rospy.init_node("motor_control_node", anonymous=True)
    pub = rospy.Publisher("/mavros/actuator_control", ActuatorControl, queue_size=10)

    rate = rospy.Rate(100)  # 50Hz 发送控制指令

    while not rospy.is_shutdown():
        msg = ActuatorControl()
        msg.header.stamp = rospy.Time.now()
        msg.group_mix = 2  # 选择"直接控制电机输出"模式
        
        # 设定四个电机的推力 (0.0 ~ 1.0)
        msg.controls[0] = 0.6  # 电机1
        msg.controls[1] = 0.6  # 电机2
        msg.controls[2] = 0.6  # 电机3
        msg.controls[3] = 0.6  # 电机4
        
        pub.publish(msg)
        rate.sleep()

def offboard_control():
    rospy.init_node("offb_node_py")

    state_sub = rospy.Subscriber("mavros/state", State, callback = state_cb)

    local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)

    rospy.wait_for_service("/mavros/cmd/arming")
    arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

    rospy.wait_for_service("/mavros/set_mode")
    set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)


    # Setpoint publishing MUST be faster than 2Hz
    rate = rospy.Rate(20)

    # Wait for Flight Controller connection
    while(not rospy.is_shutdown() and not current_state.connected):
        rate.sleep()

    pose = PoseStamped()

    pose.pose.position.x = 0
    pose.pose.position.y = 0
    pose.pose.position.z = 2

    # Send a few setpoints before starting
    for i in range(100):
        if(rospy.is_shutdown()):
            break

        local_pos_pub.publish(pose)
        rate.sleep()

    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'

    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True

    last_req = rospy.Time.now()

    while(not rospy.is_shutdown()):
        if(current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
            if(set_mode_client.call(offb_set_mode).mode_sent == True):
                rospy.loginfo("OFFBOARD enabled")

            last_req = rospy.Time.now()
        else:
            if(not current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
                if(arming_client.call(arm_cmd).success == True):
                    rospy.loginfo("Vehicle armed")

                last_req = rospy.Time.now()

        local_pos_pub.publish(pose)

        rate.sleep()

if __name__ == "__main__":
    send_body_rate_thrust()
    # send_motor_commands()
