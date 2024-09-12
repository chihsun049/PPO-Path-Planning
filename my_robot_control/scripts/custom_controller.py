#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import ApplyJointEffort
from sensor_msgs.msg import Imu, JointState
import math

class CustomController:
    def __init__(self):
        rospy.init_node('custom_controller', anonymous=True)

        # 订阅 /cmd_vel 和 /imu 话题
        self.cmd_vel_subscriber = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        self.imu_subscriber = rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        self.joint_state_subscriber = rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)

        # 定义应用关节力的服务
        rospy.wait_for_service('/gazebo/apply_joint_effort')
        self.apply_joint_effort_service = rospy.ServiceProxy('/gazebo/apply_joint_effort', ApplyJointEffort)

        # 定义关节名字（请根据你的机器人模型进行调整）
        self.left_rear_wheel_joint = 'wheel_rear_left_spin'
        self.right_rear_wheel_joint = 'wheel_rear_right_spin'
        self.front_left_steer_joint = 'wheel_front_left_steer_spin'
        self.front_right_steer_joint = 'wheel_front_right_steer_spin'

        # 保存每个转向关节的当前角度
        self.current_steering_angles = {
            self.front_left_steer_joint: 0.0,
            self.front_right_steer_joint: 0.0
        }

        # 车辆的倾斜角度
        self.pitch_angle = 0.0

        self.vehicle_mass = 12.0

        # 初始速度
        self.current_speed = 0.0

    def joint_states_callback(self, msg):
        # 保存每个转向关节的当前角度
        for i, name in enumerate(msg.name):
            if name in self.current_steering_angles:
                self.current_steering_angles[name] = msg.position[i]

    def imu_callback(self, msg):
        # 获取车辆的倾斜角度 (pitch)
        orientation_q = msg.orientation
        _, pitch, _ = self.quaternion_to_euler(orientation_q)
        self.pitch_angle = pitch

    def quaternion_to_euler(self, q):
        # 将四元数转换为欧拉角度
        t0 = 2.0 * (q.w * q.x + q.y * q.z)
        t1 = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(t0, t1)

        t2 = 2.0 * (q.w * q.y - q.z * q.x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        t3 = 2.0 * (q.w * q.z + q.x * q.y)
        t4 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(t3, t4)

        return roll, pitch, yaw

    def cmd_vel_callback(self, msg):
        # 记录当前速度
        self.current_speed = msg.linear.x

        # 调整后的控制逻辑
        gain = self.calculate_dynamic_gain(msg.linear.x, msg.angular.z)

        # 计算坡度补偿
        slope_compensation = self.calculate_slope_compensation()

        # 计算左后轮和右后轮的努力值，基于线速度、角速度和坡度补偿
        left_wheel_effort, right_wheel_effort = self.calculate_differential_drive(msg.linear.x + slope_compensation, msg.angular.z, gain)

        # 对 efforts 做限制，避免过大或过小
        max_effort = 20.0
        left_wheel_effort = max(min(left_wheel_effort, max_effort), -max_effort)
        right_wheel_effort = max(min(right_wheel_effort, max_effort), -max_effort)

        # 计算前轮转向角度
        steer_angle = msg.angular.z

        # 限制转向角度，避免过度转向
        max_steer_angle = 0.61  # 增加最大转向角度
        steer_angle = max(min(steer_angle, max_steer_angle), -max_steer_angle)

        # 应用后轮驱动
        self.apply_joint_effort(self.left_rear_wheel_joint, left_wheel_effort)
        self.apply_joint_effort(self.right_rear_wheel_joint, right_wheel_effort)

        # 应用前轮转向角度
        self.apply_steering_effort(self.front_left_steer_joint, steer_angle)
        self.apply_steering_effort(self.front_right_steer_joint, steer_angle)

    def calculate_differential_drive(self, linear_speed, angular_speed, gain):
        # 计算左右车轮的速度差
        wheel_base = 1.0  # 前后轮之间的距离，单位：米
        left_wheel_speed = linear_speed - angular_speed * wheel_base / 2.0
        right_wheel_speed = linear_speed + angular_speed * wheel_base / 2.0

        # 使用 gain 调整左右车轮的 effort
        left_wheel_effort = left_wheel_speed * gain
        right_wheel_effort = right_wheel_speed * gain

        return left_wheel_effort, right_wheel_effort

    def calculate_slope_compensation(self):
        # 根据车辆的倾斜角度计算坡度补偿
        gravity = 9.81  # 地球重力加速度 (m/s^2)
        slope_force = self.vehicle_mass * gravity * math.sin(self.pitch_angle) * 10
        
        # 根据坡度的正负，决定是增加还是减少动力
        if self.pitch_angle > 0.5:  # 上坡时增加动力
            return slope_force
        elif self.pitch_angle < -0.5:  # 下坡时减少动力
            return -slope_force
        else:
            return 0.0  # 平地或无明显坡度时，无需补偿

    def calculate_dynamic_gain(self, linear_speed, angular_z):
        # 根据线速度和转向角度调整增益
        if linear_speed < 1.1:
            if abs(angular_z) > 0.3:
                return 1.9
            else:
                return 1.7
        else:
            return 1.5

    def apply_joint_effort(self, joint_name, effort):
        duration = rospy.Duration(0.1)  # 缩短持续时间，加快响应
        try:
            self.apply_joint_effort_service(joint_name, effort, rospy.Time(0), duration)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def apply_steering_effort(self, joint_name, angle):
        # 使用从机器人获取的实际角度
        current_angle = self.current_steering_angles.get(joint_name, 0.0)
        effort = 8 * (angle - current_angle)  # 增大转向力度的比例增益
        duration = rospy.Duration(0.1)  # 缩短持续时间，加快响应
        try:
            self.apply_joint_effort_service(joint_name, effort, rospy.Time(0), duration)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def apply_braking_effort(self):
        # 在没有速度输入时施加轻微的反向力矩，防止滑动
        braking_effort = -0.1  # 轻微反向力矩
        self.apply_joint_effort(self.left_rear_wheel_joint, braking_effort)
        self.apply_joint_effort(self.right_rear_wheel_joint, braking_effort)

    def monitor_vehicle_state(self):
        rate = rospy.Rate(10)  # 以 10Hz 的频率循环
        while not rospy.is_shutdown():
            if self.current_speed == 0.0:
                self.apply_braking_effort()
            rate.sleep()

def main():
    controller = CustomController()
    rospy.sleep(1.0)  # 确保节点初始化完成
    controller.monitor_vehicle_state()  # 启动车辆状态监控
    rospy.spin()

if __name__ == '__main__':
    main()
