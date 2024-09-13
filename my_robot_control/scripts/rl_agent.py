#!/usr/bin/env python3
import rospy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import PointCloud2, Imu
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState, ContactsState
import sensor_msgs.point_cloud2 as pc2
from scipy.special import comb
from collections import namedtuple
import cv2
import open3d as o3d
import tf
from tf.transformations import quaternion_from_euler
import time
from torch.amp import GradScaler
import wandb

# Hyperparameters
REFERENCE_DISTANCE_TOLERANCE = 0.65
MEMORY_SIZE = 10000
BATCH_SIZE = 256
GAMMA = 0.99
LEARNING_RATE = 0.0003
PPO_EPOCHS = 5
CLIP_PARAM = 0.2
PREDICTION_HORIZON = 400  # MPC预测的时间步数
CONTROL_HORIZON = 10  # MPC控制的时间步数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [None] * capacity
        self.position = 0
        self.priorities = torch.zeros((capacity,), dtype=torch.float32).cuda()
        self.alpha = 0.6
        self.epsilon = 1e-5

    def add(self, state, action, reward, done, next_state):
        if state is None or action is None or reward is None or done is None or next_state is None:
            rospy.logwarn("Warning: Attempted to add None to memory, skipping entry.")
            return

        max_priority = self.priorities.max() if self.memory[self.position] is not None else torch.tensor(1.0, device=device)
        self.memory[self.position] = (
            torch.tensor(state, dtype=torch.float32, device=device),
            torch.tensor(action, dtype=torch.float32, device=device),
            torch.tensor(reward, dtype=torch.float32, device=device),
            torch.tensor(done, dtype=torch.float32, device=device),
            torch.tensor(next_state, dtype=torch.float32, device=device)
        )
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if self.position == 0:
            raise ValueError("No samples available in memory.")
        
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        if torch.isnan(priorities).any():
            priorities = torch.nan_to_num(priorities, nan=0.0)
        
        probabilities = priorities ** self.alpha
        total = probabilities.sum()

        if total > 0:
            probabilities /= total
        else:
            probabilities = torch.ones_like(probabilities) / len(probabilities)

        indices = torch.multinomial(probabilities, batch_size, replacement=False).cuda()
        
        samples = [self.memory[idx] for idx in indices if self.memory[idx] is not None]

        if len(samples) == 0 or any(sample is None for sample in samples):
            raise ValueError("Sampled None from memory.")

        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        batch = list(zip(*samples))
        states, actions, rewards, dones, next_states = batch

        return (
            torch.stack(states).to(device),
            torch.stack(actions).to(device),
            torch.stack(rewards).to(device),
            torch.stack(dones).to(device),
            torch.stack(next_states).to(device),
            indices,
            weights.to(device)
        )

    def update_priorities(self, batch_indices, batch_priorities):
        batch_priorities = torch.tensor(batch_priorities, dtype=torch.float32).cuda()
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority + self.epsilon

    def clear(self):
        self.position = 0
        self.memory = [None] * self.capacity
        self.priorities = torch.zeros((self.capacity,), dtype=torch.float32).cuda()

class MPCController:
    def __init__(self, prediction_horizon, control_horizon):
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.optimization_result = []

    def predict_future_trajectory(self, current_state, model, env):
        predicted_trajectory = []
        state = current_state.clone()
        previous_action = None

        hidden = None
        
        for _ in range(self.prediction_horizon):
            action, hidden = model.act(state, previous_action, hidden)
            action = action.cpu().data.numpy().flatten()
            next_state, _, _, _ = env.step(action)
            
            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            predicted_trajectory.append((state, action))
            
            previous_action = action
            
        return predicted_trajectory

    def optimize_control_sequence(self, predicted_trajectory, model, env):
        optimized_actions = []
        
        for i in range(self.control_horizon):
            action = predicted_trajectory[i][1]
            
            _, action_value, _ = model.evaluate(predicted_trajectory[i][0], action)
            
            if action_value > 0.5:
                optimized_actions.append(action)
            else:
                optimized_actions.append(self.mpc_fallback_action())
            
        self.optimization_result = optimized_actions
        return optimized_actions
    
    def mpc_fallback_action(self):
        fallback_action = np.array([0.0, 0.0])
        return fallback_action

class GazeboEnv:
    def __init__(self):
        rospy.init_node('gazebo_rl_agent', anonymous=True)
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.pub_imu = rospy.Publisher('/imu/data', Imu, queue_size=10)
        self.sub_scan = rospy.Subscriber('/velodyne_points', PointCloud2, self.scan_callback)
        self.sub_collision_chassis = rospy.Subscriber('/my_robot/bumper_data', ContactsState, self.collision_callback)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.listener = tf.TransformListener()
        self.action_space = 2
        self.observation_space = (4, 64, 64)
        self.state = np.zeros(self.observation_space)
        self.done = False
        self.target_x = -5.3334
        self.target_y = -0.3768
        self.target_yaw = -0.0058
        self.waypoints = self.generate_waypoints()
        self.current_waypoint_index = 0
        self.last_twist = Twist()
        self.epsilon = 0.05
        self.mpc_controller = MPCController(PREDICTION_HORIZON, CONTROL_HORIZON)
        self.collision_detected = False
        self.max_no_progress_steps = 20
        self.no_progress_steps = 0
        self.previous_distance_to_goal = float('inf')

        # Initialize lidar data as None
        self.lidar_data = None

        # 调用优化后的路径生成方法
        self.generate_optimized_waypoints()

    def generate_waypoints(self):
        waypoints = [
            (0.2206, 0.1208, -0.0053),
            (1.2812, 0.0748, -0.0212),
            (2.3472, 0.1292, -0.0327),
            (3.4053, 0.1631, -0.0474),
            (4.4468, 0.1421, -0.0347),
            (5.5032, 0.1996, -0.0198),
            (6.5372, 0.2315, -0.0122),
            (7.5948, 0.2499, -0.0177),
            (8.6607, 0.3331, -0.0262),
            (9.6811, 0.3973, -0.0256),
            (10.6847, 0.4349, -0.0277),
            (11.719, 0.4814, -0.0429),
            (12.7995, 0.5223, -0.0264),
            (13.8983, 0.515, -0.0076),
            (14.9534, 0.6193, -0.0149),
            (15.9899, 0.7217, -0.0278),
            (17.0138, 0.7653, -0.0315),
            (18.0751, 0.8058, -0.012),
            (19.0799, 0.864, -0.0036),
            (20.1383, 0.936, -0.0091),
            (21.1929, 0.9923, -0.0271),
            (22.2351, 1.0279, -0.0041),
            (23.3374, 1.1122, -0.0091),
            (24.4096, 1.1694, -0.0037),
            (25.4817, 1.2437, 0.0065),
            (26.5643, 1.3221, -0.006),
            (27.6337, 1.4294, -0.0201),
            (28.6643, 1.4471, -0.024),
            (29.6839, 1.4987, -0.0068),
            (30.7, 1.58, 0.0019),
            (31.7796, 1.6339, 0.012),
            (32.8068, 1.7283, 0.0191),
            (33.8596, 1.8004, 0.0008),
            (34.9469, 1.9665, -0.0204),
            (35.9883, 1.9812, -0.0164),
            (37.0816, 2.0237, 0.0132),
            (38.1077, 2.1291, -0.0006),
            (39.1405, 2.1418, -0.0163),
            (40.1536, 2.2273, -0.0327),
            (41.1599, 2.2473, -0.0421),
            (42.2476, 2.2927, -0.0086),
            (43.3042, 2.341, -0.0229),
            (44.4049, 2.39, -0.0256),
            (45.5091, 2.4284, -0.0391),
            (46.579, 2.5288, -0.0324),
            (47.651, 2.4926, -0.0035),
            (48.6688, 2.6072, -0.0159),
            (49.7786, 2.6338, -0.0224),
            (50.7942, 2.6644, 0.0142),
            (51.868, 2.7625, 0.0131),
            (52.9149, 2.8676, -0.0019),
            (54.0346, 2.9602, -0.017),
            (55.0855, 2.9847, 0.0511),
            (56.1474, 3.1212, 0.08),
            (57.2397, 3.2988, 0.1531),
            (58.2972, 3.5508, 0.5749),
            (59.1103, 4.1404, 1.0891),
            (59.6059, 5.1039, 1.5337),
            (59.6032, 6.2015, 1.6305),
            (59.4278, 7.212, 1.5512),
            (59.3781, 8.2782, 1.5008),
            (59.4323, 9.2866, 1.5307),
            (59.3985, 10.304, 1.5229),
            (59.3676, 11.3302, 1.5092),
            (59.3193, 12.3833, 1.493),
            (59.359, 13.4472, 1.4758),
            (59.3432, 14.4652, 1.5391),
            (59.3123, 15.479, 1.6848),
            (59.1214, 16.4917, 1.9472),
            (58.7223, 17.4568, 2.4068),
            (57.8609, 18.1061, 2.9224),
            (56.8366, 18.3103, -3.0502),
            (55.7809, 18.0938, -2.9053),
            (54.7916, 17.707, -2.9996),
            (53.7144, 17.5087, -2.9996),
            (52.6274, 17.3683, -3.0205),
            (51.6087, 17.1364, -3.0781),
            (50.5924, 17.0295, -3.0762),
            (49.5263, 16.9058, -3.0919),
            (48.4514, 16.7769, -3.1014),
            (47.3883, 16.6701, -3.0868),
            (46.3186, 16.5403, -3.0794),
            (45.3093, 16.4615, -3.0892),
            (44.263, 16.299, -3.0817),
            (43.2137, 16.1486, -3.0878),
            (42.171, 16.0501, -3.1048),
            (41.1264, 16.0245, -3.132),
            (40.171, 16.7172, 2.9319),
            (39.1264, 16.8428, 2.9626),
            (38.1122, 17.019, -2.7588),
            (37.2234, 16.5322, -2.1481),
            (36.6845, 15.6798, -2.104),
            (36.3607, 14.7064, -2.4919),
            (35.5578, 13.9947, -2.9643),
            (34.5764, 13.7466, -3.0707),
            (33.5137, 13.6068, -3.0838),
            (32.4975, 13.5031, -3.0695),
            (31.5029, 13.3368, -3.0904),
            (30.4162, 13.1925, -3.1037),
            (29.3894, 13.067, -3.123),
            (28.3181, 12.9541, -3.1333),
            (27.3195, 12.8721, 3.1318),
            (26.2852, 12.8035, 3.1391),
            (25.241, 12.6952, -3.1389),
            (24.1598, 12.6435, 3.1131),
            (23.0712, 12.5947, 3.1247),
            (21.9718, 12.5297, 3.1398),
            (20.9141, 12.4492, 3.1287),
            (19.8964, 12.3878, 3.1154),
            (18.7163, 12.32, 3.1105),
            (17.6221, 12.2928, 3.1383),
            (16.5457, 12.2855, -3.1134),
            (15.5503, 12.1534, -3.0989),
            (14.4794, 12.0462, -3.1104),
            (13.4643, 11.9637, -3.1217),
            (12.3466, 11.7943, 3.1156),
            (11.2276, 11.6071, 2.6767),
            (10.2529, 12.0711, 2.1262),
            (9.7942, 13.0066, 1.9467),
            (9.398, 13.9699, 2.2779),
            (8.6017, 14.7268, 2.9106),
            (7.4856, 14.8902, -2.8568),
            (6.5116, 14.4724, -2.864),
            (5.4626, 14.1256, -3.0697),
            (4.3911, 13.9535, -3.1087),
            (3.3139, 13.8013, 3.1244),
            (2.2967, 13.7577, 3.1353),
            (1.2165, 13.7116, 3.1319),
            (0.1864, 13.6054, -3.1129),
            (-0.9592, 13.4747, -3.132),
            (-2.0086, 13.352, 3.1106),
            (-3.0267, 13.3358, 2.8815),
            (-4.0117, 13.5304, 2.8523),
            (-5.0541, 13.8047, 2.956),
            (-6.0953, 13.9034, 3.0928),
            (-7.1116, 13.8871, 3.1398),
            (-8.152, 13.8062, 3.1392),
            (-9.195, 13.7043, 3.1361),
            (-10.2548, 13.6152, -2.8696),
            (-11.234, 13.3289, -2.4547),
            (-11.9937, 12.6211, -2.0136),
            (-12.3488, 11.6585, -1.7069),
            (-12.4231, 10.6268, -1.56),
            (-12.3353, 9.5915, -1.558),
            (-12.2405, 8.5597, -1.568),
            (-12.1454, 7.4974, -1.5839),
            (-12.0596, 6.4487, -1.5981),
            (-12.0537, 5.3613, -1.5922),
            (-12.0269, 4.2741, -1.6054),
            (-11.999, 3.2125, -1.6172),
            (-11.9454, 2.2009, -1.4409),
            (-11.7614, 1.1884, -1.1411),
            (-11.2675, 0.2385, -0.8377),
            (-10.5404, -0.58, -0.3517),
            (-9.4494, -0.8399, -0.0687),
            (-8.3965, -0.8367, 0.0701),
            (-7.3912, -0.6242, 0.0671),
            (-6.3592, -0.463, 0.0508),
            (self.target_x, self.target_y, self.target_yaw)
        ]
        return waypoints
    
    def optimize_waypoints_with_heuristics(self):
        adjusted_waypoints = []
        for i, wp in enumerate(self.waypoints):
            if i == 0 or i == len(self.waypoints) - 1:
                adjusted_waypoints.append(wp)
                continue

            wp_x, wp_y, wp_yaw = wp

            while self.is_point_near_obstacle(wp_x, wp_y):
                direction = np.arctan2(self.target_y - wp_y, self.target_x - wp_x)
                wp_x += 0.05 * np.cos(direction)
                wp_y += 0.05 * np.sin(direction)

            adjusted_waypoints.append((wp_x, wp_y, wp_yaw))
        
        self.waypoints = adjusted_waypoints
        self.current_waypoint_index = 0

    def generate_optimized_waypoints(self):
        # 只有在有LiDAR數據的情況下進行優化
        if self.lidar_data is None:
            rospy.logwarn("LiDAR data not yet available, skipping optimization.")
            return

        optimized_waypoints = []
        for i in range(len(self.waypoints) - 1):
            current_wp = self.waypoints[i]
            next_wp = self.waypoints[i + 1]

            direction = np.arctan2(next_wp[1] - current_wp[1], next_wp[0] - current_wp[0])
            new_wp_x = current_wp[0] + 0.1 * np.cos(direction)
            new_wp_y = current_wp[1] + 0.1 * np.sin(direction)
            new_wp_yaw = direction

            # 如果障礙物靠近，開始進行多方向檢查
            if self.is_point_near_obstacle(new_wp_x, new_wp_y):
                best_x, best_y = new_wp_x, new_wp_y
                min_distance_to_obstacle = float('inf')

                # 多方向檢查
                for angle_offset in np.linspace(-np.pi/2, np.pi/2, num=18):
                    adjusted_direction = direction + angle_offset
                    temp_x = current_wp[0] + 0.2 * np.cos(adjusted_direction)
                    temp_y = current_wp[1] + 0.2 * np.sin(adjusted_direction)

                    # 檢查新的點是否遠離障礙物
                    if not self.is_point_near_obstacle(temp_x, temp_y):
                        distance_to_obstacle = self.calculate_distance_to_nearest_obstacle(temp_x, temp_y)
                        if distance_to_obstacle > min_distance_to_obstacle:
                            best_x, best_y = temp_x, temp_y
                            min_distance_to_obstacle = distance_to_obstacle

                new_wp_x, new_wp_y = best_x, best_y

            optimized_waypoints.append((new_wp_x, new_wp_y, new_wp_yaw))

        optimized_waypoints.append(self.waypoints[-1])
        self.waypoints = self.smooth_waypoints(optimized_waypoints)
        self.current_waypoint_index = 0

    # 用於計算當前點與最近障礙物之間的距離
    def calculate_distance_to_nearest_obstacle(self, x, y):
        min_distance = float('inf')
        for point in pc2.read_points(self.lidar_data, field_names=("x", "y", "z"), skip_nans=True):
            distance = np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def smooth_waypoints(self, waypoints):
        waypoints_2d = [(wp[0], wp[1]) for wp in waypoints]
        smoothed_points = self.bezier_curve(waypoints_2d)
        smoothed_waypoints = [(pt[0], pt[1], waypoints[i][2]) for i, pt in enumerate(smoothed_points)]
        self.current_waypoint_index = 0
        return smoothed_waypoints
    
    def bezier_curve(self, waypoints, n_points=100):
        waypoints = np.array(waypoints)
        n = len(waypoints) - 1

        def bernstein_poly(i, n, t):
            return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

        t = np.linspace(0.0, 1.0, n_points)
        curve = np.zeros((n_points, 2))

        for i in range(n + 1):
            curve += np.outer(bernstein_poly(i, n, t), waypoints[i])

        return curve

    def is_point_near_obstacle(self, x, y, threshold=0.25):
        if self.lidar_data is None:
            rospy.logwarn("LiDAR data is not available yet.")
            return False

        min_distance_to_obstacle = float('inf')
        for point in pc2.read_points(self.lidar_data, field_names=("x", "y", "z"), skip_nans=True):
            distance = np.sqrt((point[0] - x)**2 + (point[1] - y)**2)
            if distance < min_distance_to_obstacle:
                min_distance_to_obstacle = distance

        return min_distance_to_obstacle < threshold

    def collision_callback(self, data):
        if len(data.states) > 0:
            self.collision_detected = True
            rospy.loginfo("Collision detected!")
        else:
            self.collision_detected = False

    def is_collision_detected(self):
        return self.collision_detected

    def generate_imu_data(self):
        imu_data = Imu()
        imu_data.header.stamp = rospy.Time.now()
        imu_data.header.frame_id = 'chassis'

        imu_data.linear_acceleration.x = np.random.normal(0, 0.1)
        imu_data.linear_acceleration.y = np.random.normal(0, 0.1)
        imu_data.linear_acceleration.z = np.random.normal(9.81, 0.1)
        
        imu_data.angular_velocity.x = np.random.normal(0, 0.01)
        imu_data.angular_velocity.y = np.random.normal(0, 0.01)
        imu_data.angular_velocity.z = np.random.normal(0, 0.01)

        robot_x, robot_y, robot_yaw = self.get_robot_position()
        quaternion = quaternion_from_euler(0.0, 0.0, robot_yaw)
        imu_data.orientation.x = quaternion[0]
        imu_data.orientation.y = quaternion[1]
        imu_data.orientation.z = quaternion[2]
        imu_data.orientation.w = quaternion[3]

        return imu_data

    def scan_callback(self, data):
        if data is not None and self.is_valid_data(data):
            self.lidar_data = data
            combined_pcl_data = self.accumulate_lidar_data([data])

            if self.current_waypoint_index >= len(self.waypoints):
                rospy.logwarn("current_waypoint_index out of range, resetting to last valid index.")
                self.current_waypoint_index = len(self.waypoints) - 1

            current_waypoint_x, current_waypoint_y, current_waypoint_yaw = self.waypoints[self.current_waypoint_index]
            self.state = self.generate_occupancy_grid(combined_pcl_data, current_waypoint_x, current_waypoint_y, current_waypoint_yaw)
        else:
            rospy.logwarn("Received invalid or empty LiDAR data")

    def is_valid_data(self, data):
        for point in pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True):
            if point[0] != 0.0 or point[1] != 0.0 or point[2] != 0.0:
                return True
        return False

    def accumulate_lidar_data(self, lidar_buffer):
        combined_points = []
        for data in lidar_buffer:
            points_list = []
            for point in pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True):
                if point[0] == 0.0 and point[1] == 0.0 and point[2] == 0.0:
                    continue

                point_transformed = self.transform_point(point, 'my_robot/velodyne', 'chassis')
                points_list.append(point_transformed)
            combined_points.extend(points_list)

        if len(combined_points) == 0:
            rospy.logwarn("No points collected after transformation")
            return PointCloud2()

        combined_pcl = o3d.geometry.PointCloud()
        combined_pcl.points = o3d.utility.Vector3dVector(np.array(combined_points, dtype=np.float32))

        return self.convert_open3d_to_ros(combined_pcl)

    def transform_point(self, point, from_frame, to_frame):
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform(to_frame, from_frame, now, rospy.Duration(1.0))
            
            point_stamped = PointStamped()
            point_stamped.header.frame_id = from_frame
            point_stamped.header.stamp = now
            point_stamped.point.x = point[0]
            point_stamped.point.y = point[1]
            point_stamped.point.z = point[2]
            
            point_transformed = self.listener.transformPoint(to_frame, point_stamped)
            return [point_transformed.point.x, point_transformed.point.y, point_transformed.point.z]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Transform failed: {e}")
            return [point[0], point[1], point[2]]

    def convert_open3d_to_ros(self, cloud):
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'velodyne'
        points = np.asarray(cloud.points)
        return pc2.create_cloud_xyz32(header, points)

    def generate_occupancy_grid(self, lidar_data, waypoint_x, waypoint_y, waypoint_yaw, grid_size=0.05, map_size=100):
        if lidar_data is None or not isinstance(lidar_data, PointCloud2):
            rospy.logwarn("No LiDAR data available, returning empty occupancy grid.")
            return np.zeros((4, 64, 64))

        grid_size = float(grid_size)
        map_size_in_grids = int(map_size / grid_size)
        grid = np.zeros((map_size_in_grids, map_size_in_grids))

        x_min, x_max = -map_size / 2, map_size / 2
        y_min, y_max = -map_size / 2, map_size / 2

        for point in pc2.read_points(lidar_data, field_names=("x", "y", "z"), skip_nans=True):
            x = point[0]
            y = point[1]

            if x_min <= x <= x_max and y_min <= y <= y_max:
                ix = int((x - x_min) / grid_size)
                iy = int((y - y_min) / grid_size)

                if 0 <= ix < grid.shape[1] and 0 <= iy < grid.shape[0]:
                    grid[iy, ix] = 1
                else:
                    rospy.logwarn(f"Index out of bounds after filtering: ix={ix}, iy={iy}")
            else:
                rospy.logwarn(f"LiDAR point out of bounds after filtering: x={x}, y={y}")

        grid = cv2.resize(grid, (64, 64), interpolation=cv2.INTER_LINEAR)

        waypoint_grid = np.zeros((3, 64, 64))
        waypoint_grid[0, :, :] = waypoint_x
        waypoint_grid[1, :, :] = waypoint_y
        waypoint_grid[2, :, :] = waypoint_yaw

        occupancy_grid = np.vstack([grid[np.newaxis, :, :], waypoint_grid])

        return occupancy_grid

    def step(self, action):
        if self.current_waypoint_index >= len(self.waypoints) - 1:
            self.done = True
            return self.state, 100, self.done, {}

        if self.is_collision_detected():
            rospy.loginfo("Collision detected, resetting environment.")
            reward = -200.0  # 碰撞懲罰
            self.reset()  # 重置環境
            return self.state, reward, True, {}

        robot_x, robot_y, _ = self.get_robot_position()
        current_waypoint_x, current_waypoint_y, current_waypoint_yaw = self.waypoints[self.current_waypoint_index]
        
        # 計算與當前目標的距離
        distance_to_current_waypoint = np.sqrt((current_waypoint_x - robot_x) ** 2 + (current_waypoint_y - robot_y) ** 2)

        # 如果距離與上一步差異很小，增加無進展計數器
        if np.abs(distance_to_current_waypoint - self.previous_distance_to_goal) < 0.01:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0  # 如果有進展，重置計數器

        self.previous_distance_to_goal = distance_to_current_waypoint
        
        # 如果無進展的步數超過限制，重置環境
        if self.no_progress_steps >= self.max_no_progress_steps:
            rospy.loginfo("Robot stuck for too long, resetting environment.")
            self.reset()
            return self.state, -100.0, True, {}  # 給予懲罰並重置

        # 檢查是否接近當前路徑點
        if distance_to_current_waypoint < REFERENCE_DISTANCE_TOLERANCE:
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(self.waypoints):
                self.done = True
                return self.state, 100, self.done, {}

        # 確保 action 是一個數組或張量，並包含至少兩個元素（線速度和角速度）
        if isinstance(action, (int, float)):
            action = np.array([action, 0.0])  # 如果只有線速度，補充角速度為 0

        action = self.calculate_action_ucb(current_waypoint_x, current_waypoint_y, current_waypoint_yaw)

        # 提取線速度和角速度，並進行限制
        linear_speed = np.clip(action[0], -2.0, 2.0)
        steer_angle = np.clip(action[1], -0.6, 0.6)

        # 發布控制指令
        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = steer_angle

        self.last_twist = twist
        self.pub_cmd_vel.publish(twist)

        # 發布 IMU 數據
        imu_data = self.generate_imu_data()
        self.pub_imu.publish(imu_data)

        rospy.sleep(0.1)

        # 計算獎勵
        reward = self.calculate_reward(current_waypoint_x, current_waypoint_y)

        return self.state, reward, self.done, {}

    def reset(self):
        yaw = -0.0053
        quaternion = quaternion_from_euler(0.0, 0.0, yaw)
        state_msg = ModelState()
        state_msg.model_name = 'my_robot'
        state_msg.pose.position.x = 0.2206
        state_msg.pose.position.y = 0.1208
        state_msg.pose.position.z = 2.2
        state_msg.pose.orientation.x = quaternion[0]
        state_msg.pose.orientation.y = quaternion[1]
        state_msg.pose.orientation.z = quaternion[2]
        state_msg.pose.orientation.w = quaternion[3]

        state_msg.twist.linear.x = 0.0
        state_msg.twist.linear.y = 0.0
        state_msg.twist.linear.z = 0.0
        state_msg.twist.angular.x = 0.0
        state_msg.twist.angular.y = 0.0
        state_msg.twist.angular.z = 0.0

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.set_model_state(state_msg)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

        rospy.sleep(0.5)

        self.waypoints = self.generate_waypoints()
        self.current_waypoint_index = 0
        self.done = False

        empty_lidar_data = PointCloud2()
        current_waypoint_x, current_waypoint_y, current_waypoint_yaw = self.waypoints[self.current_waypoint_index]
        self.state = self.generate_occupancy_grid(empty_lidar_data, current_waypoint_x, current_waypoint_y, current_waypoint_yaw)

        # 停止机器人运动
        self.last_twist = Twist()
        self.pub_cmd_vel.publish(self.last_twist)  # 停止机器人

        # 重置IMU数据
        imu_data = self.generate_imu_data()
        self.pub_imu.publish(imu_data)

        self.previous_yaw_error = 0
        self.no_progress_steps = 0
        self.previous_distance_to_goal = float('inf')
        self.collision_detected = False  # 重置碰撞标志

        return self.state

    def calculate_reward(self, target_x, target_y):
        robot_x, robot_y, robot_yaw = self.get_robot_position()
        reward = 0

        # 目標方向與當前朝向的差距，鼓勵平滑的方向變化
        direction_to_target = np.arctan2(target_y - robot_y, target_x - robot_x)
        yaw_diff = np.abs(direction_to_target - robot_yaw)
        reward += max(0, 5 - yaw_diff * 5)  # 鼓勵角度偏差越小獎勵越多

        # 距離目標的獎勵，鼓勵靠近目標
        distance_to_goal = np.sqrt((target_x - robot_x) ** 2 + (target_y - robot_y) ** 2)
        reward += (1.0 / (distance_to_goal + 1e-5)) * 10  # 獎勵越靠近目標越高

        # 安全性：遠離障礙物的獎勵
        if not self.is_point_near_obstacle(robot_x, robot_y, threshold=0.3):
            reward += 50  # 如果離障礙物足夠遠，給予額外獎勵

        # 行徑方向與目標方向的差距，反向時不給任何獎勵
        robot_velocity_x = self.last_twist.linear.x
        robot_velocity_y = self.last_twist.linear.y
        if robot_velocity_x != 0 or robot_velocity_y != 0:
            movement_direction = np.arctan2(robot_velocity_y, robot_velocity_x)
            movement_diff = np.abs(movement_direction - direction_to_target)

            # 如果角度差小於90度，給予獎勵
            if movement_diff < np.pi / 2:  # 小於90度才給獎勵
                if movement_diff < np.pi / 6:  # 30度內認為是朝向目標
                    reward += 20  # 額外獎勵朝向目標行徑
            else:
                reward = 0  # 逆向時，所有獎勵歸零

        # 獎勵平滑行駛，根據上一個動作來計算加速度變化，避免突然加速或轉向
        linear_speed = np.linalg.norm([self.last_twist.linear.x, self.last_twist.linear.y])
        acceleration = np.abs(self.last_twist.linear.x - linear_speed)
        if acceleration < 0.1:
            reward += 10  # 獎勵平穩行駛

        return reward

    def get_robot_position(self):
        try:
            rospy.wait_for_service('/gazebo/get_model_state')
            model_state = self.get_model_state('my_robot', '')
            robot_x = model_state.pose.position.x
            robot_y = model_state.pose.position.y

            orientation_q = model_state.pose.orientation
            yaw = self.quaternion_to_yaw(orientation_q)
            return robot_x, robot_y, yaw
        except rospy.ServiceException as e:
            rospy.logerr(f"Get model state service call failed: %s", e)
            return 0, 0, 0

    def quaternion_to_yaw(self, orientation_q):
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return yaw

    def calculate_action_ucb(self, waypoint_x, waypoint_y, waypoint_yaw):
        robot_x, robot_y, robot_yaw = self.get_robot_position()
        linear_speed = np.linalg.norm([self.last_twist.linear.x, self.last_twist.linear.y])

        preview_distance = 1.2  # 預覽距離
        num_future_points = 7  # 預覽點的數量
        future_yaw_errors = []
        future_turn_detected = False

        # 檢查是否已經超過當前路徑點，並切換到下一個
        distance_to_current_waypoint = np.sqrt((waypoint_x - robot_x) ** 2 + (waypoint_y - robot_y) ** 2)
        if distance_to_current_waypoint < REFERENCE_DISTANCE_TOLERANCE:
            self.current_waypoint_index += 1  # 切換到下一個路徑點
            if self.current_waypoint_index >= len(self.waypoints):
                rospy.loginfo("已經到達最後一個路徑點")
                self.done = True
                return np.array([0.0, 0.0])  # 到達最後一個路徑點，停止動作
            waypoint_x, waypoint_y, waypoint_yaw = self.waypoints[self.current_waypoint_index]
        
        for i in range(1, num_future_points + 1):
            future_preview_distance = preview_distance * i
            future_waypoint_x = waypoint_x + future_preview_distance * np.cos(waypoint_yaw)
            future_waypoint_y = waypoint_y + future_preview_distance * np.sin(waypoint_yaw)

            future_direction_to_waypoint = np.arctan2(future_waypoint_y - robot_y, future_waypoint_x - robot_x)
            yaw_error = future_direction_to_waypoint - robot_yaw
            yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

            future_yaw_errors.append(yaw_error)

            # 如果未來某點的 yaw 誤差超過閾值，標記為即將轉彎
            if np.abs(yaw_error) > 0.3:  # 假設 0.3 為轉彎的閾值
                future_turn_detected = True

        total_yaw_error = sum(future_yaw_errors)
        average_yaw_error = total_yaw_error / num_future_points

        # 如果即將轉彎，提前降低速度
        if future_turn_detected:
            linear_speed = max(0.5, linear_speed * 0.5)  # 提前減速到過彎速度
        else:
            # 沒有轉彎時，正常調整速度
            if average_yaw_error > 0.3:
                linear_speed = max(1.0, linear_speed * 0.7)
            elif 0.1 < average_yaw_error <= 0.3:
                linear_speed = max(1.5, linear_speed * 0.9)
            else:
                linear_speed = min(2.0, linear_speed * 1.1)  # 平直路徑上加速

        # 調整控制參數 kp 和 kd
        if linear_speed < 1.0:
            kp = 0.5
            kd = 0.2
        elif linear_speed < 1.5:
            kp = 0.4
            kd = 0.3
        else:
            kp = 0.3
            kd = 0.4

        previous_yaw_error = getattr(self, 'previous_yaw_error', 0)
        current_yaw_error_rate = future_yaw_errors[0] - previous_yaw_error

        # 計算方向盤角度
        steer_angle = kp * future_yaw_errors[0] + kd * current_yaw_error_rate
        steer_angle = np.clip(steer_angle, -0.6, 0.6)

        self.previous_yaw_error = future_yaw_errors[0]

        action = np.array([linear_speed, steer_angle])
        return action

class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)  # 增加一层卷积层

        self._to_linear = self._get_conv_output_size(observation_space)
        
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True)  # 增加LSTM层数
        self.fc2 = nn.Linear(128, 128)
        
        self.actor = nn.Linear(128, action_space)
        self.critic = nn.Linear(128, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_space))

        # Initialize UCB variables
        self.visit_counts = {}  # Dictionary to store visit counts for each (state, action)
        self.c = 2  # UCB exploration coefficient (can be adjusted)

    def _get_conv_output_size(self, shape):
        x = torch.zeros(1, *shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # 通过额外的卷积层
        return int(np.prod(x.size()))

    def forward(self, x, hidden=None):
        if len(x.shape) == 5:
            x = x.squeeze(1)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))  # 通过额外的卷积层
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))

        if hidden is None:
            x, hidden = self.lstm(x.unsqueeze(0))
        else:
            x, hidden = self.lstm(x.unsqueeze(0), hidden)

        x = torch.relu(self.fc2(x.squeeze(0)))
        
        action_mean = self.actor(x)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        value = self.critic(x)

        return action_mean, action_std, value, hidden

    def act(self, state, previous_action=None, hidden=None):
        with torch.amp.autocast('cuda'):
            action_mean, action_std, _, hidden = self(state, hidden)

        # Upper Confidence Bound (UCB) for action selection
        state_key = tuple(state.cpu().numpy().flatten())
        
        # Ensure the visit count is initialized to match the size of action_space, not action_mean
        if state_key not in self.visit_counts:
            # Initialize visit counts to match the action_space size, assuming action_space is a scalar
            action_space_size = action_mean.shape[-1]  # Get the action space dimension
            self.visit_counts[state_key] = torch.zeros(action_space_size, device=device)  # Ensure it's size of the action space

        # Convert total_state_visits to Tensor
        total_state_visits = self.visit_counts[state_key].sum()
        total_state_visits = torch.tensor(total_state_visits).to(device)  # Ensure it's a Tensor
        
        if total_state_visits == 0:
            total_state_visits = torch.tensor(1.0).to(device)  # avoid division by zero

        # UCB adjustment
        ucb_bonus = self.c * torch.sqrt(torch.log(total_state_visits) / (self.visit_counts[state_key] + 1e-5))
        adjusted_action_mean = action_mean + ucb_bonus

        # Choose action with added exploration from UCB
        action = adjusted_action_mean + action_std * torch.randn_like(action_std)

        # Update the visit counts for chosen action
        chosen_action = action.argmax().item()  # Ensure chosen_action is an integer index
        self.visit_counts[state_key][chosen_action] += 1  # Increment the visit count for the chosen action

        return action, hidden
    
    def evaluate(self, state, action):
        action_mean, action_std, value, hidden = self(state)
        
        if action.dim() == 1:
            action = action.unsqueeze(1)
        
        action = torch.tensor(action, dtype=torch.float32).to(action_mean.device)

        action_log_probs = -((action - action_mean) ** 2) / (2 * action_std ** 2) - action_std.log() - 0.5 * torch.log(torch.tensor(2 * np.pi, device=action_mean.device))
        action_log_probs = action_log_probs.sum(1)

        return action_log_probs, value, hidden

def ppo_update(ppo_epochs, env, model, optimizer, memory, scaler):
    for _ in range(ppo_epochs):
        state_batch, action_batch, reward_batch, done_batch, next_state_batch, indices, weights = memory.sample(BATCH_SIZE)
        
        adjusted_lr = LEARNING_RATE * (weights.mean().item() + 1e-3)
        for param_group in optimizer.param_groups:
            param_group['lr'] = adjusted_lr
            
        state_batch = torch.tensor(state_batch, dtype=torch.float32).to(device)
        action_batch = torch.tensor(action_batch, dtype=torch.float32).to(device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).view(-1, 1).to(device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32).view(-1, 1).to(device)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).view(-1, 1).to(device)
        
        with torch.no_grad():
            old_log_probs, _, _ = model.evaluate(state_batch, action_batch)
        old_log_probs = old_log_probs.detach()

        for _ in range(PPO_EPOCHS):
            with torch.amp.autocast('cuda'):
                log_probs, state_values, optimized_paths = model.evaluate(state_batch, action_batch)
                advantages = reward_batch + (1 - done_batch) * GAMMA * model(next_state_batch)[2].detach() - state_values

                ratio = (log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_PARAM, 1 + CLIP_PARAM) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(state_values, reward_batch + (1 - done_batch) * GAMMA * model(next_state_batch)[2].detach())
                loss = actor_loss + 0.5 * critic_loss


            # wandb.log({
            #     "actor_loss": actor_loss.item(),
            #     "critic_loss": critic_loss.item(),
            #     "total_loss": loss.item()
            # })
            
            scaler.scale(loss).backward()
            if _ % BATCH_SIZE == 0:
                scaler.step(optimizer)
                scaler.update()

            priorities = (advantages + 1e-5).abs().detach().cpu().numpy()
            memory.update_priorities(indices, priorities)

def main():
    env = GazeboEnv()
    model = ActorCritic(env.observation_space, env.action_space).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler('cuda')
    memory = PrioritizedMemory(MEMORY_SIZE)

    model_path = "/home/chihsun/catkin_ws/src/my_robot_control/scripts/saved_model_ppo.pth"
    best_model_path = "/home/chihsun/catkin_ws/src/my_robot_control/scripts/best_model.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded existing model.")
    else:
        print("Created new model.")

    # wandb.init(project="my_robot_workspace")

    # wandb.config = {
    #     "learning_rate": LEARNING_RATE,
    #     "batch_size": BATCH_SIZE,
    #     "gamma": GAMMA,
    #     "ppo_epochs": PPO_EPOCHS,
    #     "clip_param": CLIP_PARAM,
    #     "memory_size": MEMORY_SIZE,
    #     "prediction_horizon": PREDICTION_HORIZON,
    #     "control_horizon": CONTROL_HORIZON,
    # }

    num_episodes = 1000000
    best_test_reward = -np.inf

    for e in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        previous_action = None
        hidden = None
        total_reward = 0

        start_time = time.time()

        for time_step in range(1500):
            action_values, hidden = model.act(state, previous_action, hidden)
            action = action_values.argmax().item()
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)

            memory.add(state.cpu().numpy(), action, reward, done, next_state.cpu().numpy())

            state = next_state
            previous_action = action
            total_reward += reward
            # print("reward++")

            elapsed_time = time.time() - start_time

            if done or elapsed_time > 240:
                if elapsed_time > 240:
                    reward -= 1000.0
                    print(f"Episode {e} failed at time step {time_step}: time exceeded 240 sec.")
                break

        ppo_update(PPO_EPOCHS, env, model, optimizer, memory, scaler)
        memory.clear()

        print(f"Episode {e}, Total Reward: {total_reward}")

        # 優化路徑點
        env.generate_optimized_waypoints()

        # # 记录到 wandb
        # wandb.log({
        #     "episode": e,
        #     "total_reward": total_reward,
        #     "best_test_reward": best_test_reward,
        #     "elapsed_time": elapsed_time
        # })

        if total_reward > best_test_reward:
            best_test_reward = total_reward
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with reward: {best_test_reward}")

        if e % 20 == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved after {e} episodes.")

        rospy.sleep(1.0)

    torch.save(model.state_dict(), model_path)
    print("Final model saved.")
    # wandb.finish()

if __name__ == '__main__':
    main()
