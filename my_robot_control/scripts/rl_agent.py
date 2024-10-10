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

# 超參數
REFERENCE_DISTANCE_TOLERANCE = 0.65
MEMORY_SIZE = 10000
BATCH_SIZE = 256
GAMMA = 0.99
LEARNING_RATE = 0.0003
PPO_EPOCHS = 10
CLIP_PARAM = 0.2
PREDICTION_HORIZON = 400
CONTROL_HORIZON = 10

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
        # 確保每個 priority 是單一標量
        for idx, priority in zip(batch_indices, batch_priorities):
            # 如果 priority 是 numpy 陣列，檢查其 size
            if priority.size > 1:
                priority = priority[0]
            self.priorities[idx] = priority.item() + self.epsilon

    def clear(self):
        self.position = 0
        self.memory = [None] * self.capacity
        self.priorities = torch.zeros((self.capacity,), dtype=torch.float32).cuda()

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
        self.observation_space = (3, 64, 64)
        self.state = np.zeros(self.observation_space)
        self.done = False
        self.target_x = -5.3334
        self.target_y = -0.3768
        self.waypoints = self.generate_waypoints()
        self.current_waypoint_index = 0
        self.last_twist = Twist()
        self.epsilon = 0.05
        self.collision_detected = False
        self.previous_robot_position = None  # 初始化 previous_robot_position 為 None
        self.previous_distance_to_goal = None  # 初始化 previous_distance_to_goal 為 None

        self.lidar_data = None  # 初始化 lidar_data，避免 AttributeError

        self.max_no_progress_steps = 10
        self.no_progress_steps = 0
        self.optimized_waypoints = []

    def generate_waypoints(self):
        waypoints = [
            (0.2206, 0.1208),
            (1.2812, 0.0748),
            (2.3472, 0.129),
            (3.4053, 0.1631),
            (4.4468, 0.1421),
            (5.5032, 0.1996),
            (6.5372, 0.2315),
            (7.5948, 0.2499),
            (8.6607, 0.3331),
            (9.6811, 0.3973),
            (10.6847, 0.4349),
            (11.719, 0.4814),
            (12.7995, 0.5223),
            (13.8983, 0.515),
            (14.9534, 0.6193),
            (15.9899, 0.7217),
            (17.0138, 0.7653),
            (18.0751, 0.8058),
            (19.0799, 0.864),
            (20.1383, 0.936),
            (21.1929, 0.9923),
            (22.2351, 1.0279),
            (23.3374, 1.1122),
            (24.4096, 1.1694),
            (25.4817, 1.2437),
            (26.5643, 1.3221),
            (27.6337, 1.4294),
            (28.6643, 1.4471),
            (29.6839, 1.4987),
            (30.7, 1.58),
            (31.7796, 1.6339),
            (32.8068, 1.7283),
            (33.8596, 1.8004),
            (34.9469, 1.9665),
            (35.9883, 1.9812),
            (37.0816, 2.0237),
            (38.1077, 2.1291),
            (39.1405, 2.1418),
            (40.1536, 2.2273),
            (41.1599, 2.2473),
            (42.2476, 2.2927),
            (43.3042, 2.341),
            (44.4049, 2.39),
            (45.5091, 2.4284),
            (46.579, 2.5288),
            (47.651, 2.4926),
            (48.6688, 2.6072),
            (49.7786, 2.6338),
            (50.7942, 2.6644),
            (51.868, 2.7625),
            (52.9149, 2.8676),
            (54.0346, 2.9602),
            (55.0855, 2.9847),
            (56.1474, 3.1212),
            (57.2397, 3.2988),
            (58.2972, 3.5508),
            (59.1103, 4.1404),
            (59.6059, 5.1039),
            (59.6032, 6.2015),
            (59.4278, 7.212),
            (59.3781, 8.2782),
            (59.4323, 9.2866),
            (59.3985, 10.304),
            (59.3676, 11.3302),
            (59.3193, 12.3833),
            (59.359, 13.4472),
            (59.3432, 14.4652),
            (59.3123, 15.479),
            (59.1214, 16.4917),
            (58.7223, 17.4568),
            (57.8609, 18.1061),
            (56.8366, 18.3103),
            (55.7809, 18.0938),
            (54.7916, 17.707),
            (53.7144, 17.5087),
            (52.6274, 17.3683),
            (51.6087, 17.1364),
            (50.5924, 17.0295),
            (49.5263, 16.9058),
            (48.4514, 16.7769),
            (47.3883, 16.6701),
            (46.3186, 16.5403),
            (45.3093, 16.4615),
            (44.263, 16.299),
            (43.2137, 16.1486),
            (42.171, 16.0501),
            (41.1264, 16.0245),
            (40.171, 16.7172),
            (39.1264, 16.8428),
            (38.1122, 17.019),
            (37.2234, 16.5322),
            (36.6845, 15.6798),
            (36.3607, 14.7064),
            (35.5578, 13.9947),
            (34.5764, 13.7466),
            (33.5137, 13.6068),
            (32.4975, 13.5031),
            (31.5029, 13.3368),
            (30.4162, 13.1925),
            (29.3894, 13.067),
            (28.3181, 12.9541),
            (27.3195, 12.8721),
            (26.2852, 12.8035),
            (25.241, 12.6952),
            (24.1598, 12.6435),
            (23.0712, 12.5947),
            (21.9718, 12.5297),
            (20.9141, 12.4492),
            (19.8964, 12.3878),
            (18.7163, 12.32),
            (17.6221, 12.2928),
            (16.5457, 12.2855),
            (15.5503, 12.1534),
            (14.4794, 12.0462),
            (13.4643, 11.9637),
            (12.3466, 11.7943),
            (11.2276, 11.6071),
            (10.2529, 12.0711),
            (9.7942, 13.0066),
            (9.398, 13.9699),
            (8.6017, 14.7268),
            (7.4856, 14.8902),
            (6.5116, 14.4724),
            (5.4626, 14.1256),
            (4.3911, 13.9535),
            (3.3139, 13.8013),
            (2.2967, 13.7577),
            (1.2165, 13.7116),
            (0.1864, 13.6054),
            (-0.9592, 13.4747),
            (-2.0086, 13.352),
            (-3.0267, 13.3358),
            (-4.0117, 13.5304),
            (-5.0541, 13.8047),
            (-6.0953, 13.9034),
            (-7.1116, 13.8871),
            (-8.152, 13.8062),
            (-9.195, 13.7043),
            (-10.2548, 13.6152),
            (-11.234, 13.3289),
            (-11.9937, 12.6211),
            (-12.3488, 11.6585),
            (-12.4231, 10.6268),
            (-12.3353, 9.5915),
            (-12.2405, 8.5597),
            (-12.1454, 7.4974),
            (-12.0596, 6.4487),
            (-12.0537, 5.3613),
            (-12.0269, 4.2741),
            (-11.999, 3.2125),
            (-11.9454, 2.2009),
            (-11.7614, 1.1884),
            (-11.2675, 0.2385),
            (-10.5404, -0.58),
            (-9.4494, -0.8399),
            (-8.3965, -0.8367),
            (-7.3912, -0.6242),
            (-6.3592, -0.463),
            (self.target_x, self.target_y)
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

    def generate_optimized_waypoints(self, optimization_interval=10, safety_distance=1.0):
        if self.lidar_data is None:
            rospy.logwarn("LiDAR data not yet available, skipping optimization.")
            return

        optimized_waypoints = []
        for i in range(len(self.waypoints) - 1):
            current_wp = self.waypoints[i]
            next_wp = self.waypoints[i + 1]

            # 計算方向和法線向量
            direction = np.arctan2(next_wp[1] - current_wp[1], next_wp[0] - current_wp[0])
            normal_direction = direction + np.pi / 2  # 法線向量方向

            # 初始新路徑點沿著方向前進
            new_wp_x = current_wp[0] + 0.1 * np.cos(direction)
            new_wp_y = current_wp[1] + 0.1 * np.sin(direction)

            # 加入優化間隔控制
            if i % optimization_interval == 0 or self.is_point_near_obstacle(new_wp_x, new_wp_y, safety_distance):
                best_x, best_y = new_wp_x, new_wp_y
                max_distance_to_obstacle = 0

                # 沿著法線方向偏移並嘗試最佳化
                for angle_offset in [-np.pi / 4, 0, np.pi / 4]:  # 左、中、右三方向
                    adjusted_direction = normal_direction + angle_offset
                    for offset_distance in [0.1, 0.2, 0.3]:
                        temp_x = new_wp_x + offset_distance * np.cos(adjusted_direction)
                        temp_y = new_wp_y + offset_distance * np.sin(adjusted_direction)

                        # 避免盲區及檢查安全距離
                        if not self.is_point_near_obstacle(temp_x, temp_y) and not self.is_in_blind_spot(temp_x, temp_y):
                            distance_to_obstacle = self.calculate_distance_to_nearest_obstacle(temp_x, temp_y)
                            if distance_to_obstacle > max_distance_to_obstacle:
                                best_x, best_y = temp_x, temp_y
                                max_distance_to_obstacle = distance_to_obstacle

                # 使用最佳點作為新的路徑點
                new_wp_x, new_wp_y = best_x, best_y

            optimized_waypoints.append((new_wp_x, new_wp_y))

        # 添加最終目標點
        optimized_waypoints.append(self.waypoints[-1])
        self.waypoints = self.smooth_waypoints(optimized_waypoints)  # 平滑路徑
        self.current_waypoint_index = 0  # 重置路徑索引

    def is_point_near_obstacle(self, x, y, threshold=1.2):
        """
        確保安全距離，檢查是否靠近障礙物
        """
        if self.lidar_data is None:
            rospy.logwarn("LiDAR data is not available yet.")
            return False

        min_distance_to_obstacle = float('inf')
        for point in pc2.read_points(self.lidar_data, field_names=("x", "y", "z"), skip_nans=True):
            distance = np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2)
            if distance < min_distance_to_obstacle:
                min_distance_to_obstacle = distance

        # 判斷是否在安全閾值內
        return min_distance_to_obstacle < threshold or self.is_in_blind_spot(x, y)

    def is_in_blind_spot(self, x, y, blind_spot_length=3.36):
        """
        檢查給定的點是否位於車輛的盲區內
        """
        blind_spot_edges = self.calculate_blind_spot_edge_distances()
        min_distance_to_edges = min(blind_spot_edges)

        # 如果距離小於盲區長度，則該點在盲區內
        return min_distance_to_edges < blind_spot_length

    # 用於計算當前點與最近障礙物之間的距離
    def calculate_distance_to_nearest_obstacle(self, x, y):
        min_distance = float('inf')
        for point in pc2.read_points(self.lidar_data, field_names=("x", "y", "z"), skip_nans=True):
            distance = np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def smooth_waypoints(self, waypoints):
        waypoints_2d = [(wp[0], wp[1]) for wp in waypoints]  # Only keep x and y coordinates
        smoothed_points = self.bezier_curve(waypoints_2d)
        smoothed_waypoints = [(pt[0], pt[1]) for pt in smoothed_points]  # No yaw involved
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
    
    def calculate_blind_spot_edge_distances(self):
        # 计算车辆的边缘距离，同时考虑盲区
        blind_spot_length = 3.36  # 根据上面的计算结果，盲区长度为 3.36 米
        vehicle_edges = [
            (1.0, 0),    # 前方边缘
            (-1.0, 0),   # 后方边缘
            (0, 0.5),    # 左侧边缘
            (0, -0.5)    # 右侧边缘
        ]
        
        edge_distances = []
        for edge in vehicle_edges:
            edge_x, edge_y = edge
            distance = self.calculate_distance_to_nearest_obstacle(edge_x, edge_y)
            if distance < blind_spot_length:
                distance = blind_spot_length  # 如果障碍物在盲区内，设定为盲区距离
            edge_distances.append(distance)
        
        return edge_distances

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
        # 确保数据有效
        if data is not None and self.is_valid_data(data):
            self.lidar_data = data
            combined_pcl_data = self.accumulate_lidar_data([data])

            if self.current_waypoint_index >= len(self.waypoints):
                rospy.logwarn("current_waypoint_index out of range, resetting to last valid index.")
                self.current_waypoint_index = len(self.waypoints) - 1

            current_waypoint_x, current_waypoint_y = self.waypoints[self.current_waypoint_index]

            self.state = self.generate_occupancy_grid(combined_pcl_data, current_waypoint_x, current_waypoint_y)

            self.future_waypoints = self.waypoints[self.current_waypoint_index+1 : min(self.current_waypoint_index+1+7, len(self.waypoints))]

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

    def generate_occupancy_grid(self, lidar_data, waypoint_x, waypoint_y, grid_size=0.05, map_size=100):
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

        waypoint_grid = np.zeros((2, 64, 64))
        waypoint_grid[0, :, :] = waypoint_x
        waypoint_grid[1, :, :] = waypoint_y

        occupancy_grid = np.vstack([grid[np.newaxis, :, :], waypoint_grid])

        return occupancy_grid

    def step(self, action):
        reward = 0  # 初始化 reward
        # 取得當前機器人的位置
        robot_x, robot_y, robot_yaw = self.get_robot_position()
        current_waypoint_x, current_waypoint_y = self.waypoints[self.current_waypoint_index]

        # 判斷是否到達最終目標點
        if np.linalg.norm([robot_x - self.target_x, robot_y - self.target_y]) < 0.5:
            self.done = True
            reward += 1000  # 給予較高的獎勵表示成功到達終點

        # 檢查是否發生碰撞
        if self.is_collision_detected():
            rospy.loginfo("Collision detected, resetting environment.")
            reward = -200.0  # 碰撞懲罰
            self.reset()  # 重置環境
            return self.state, reward, True, {}

        # 計算與目標路徑點的距離
        distance_to_goal = np.linalg.norm([current_waypoint_x - robot_x, current_waypoint_y - robot_y])

        # 如果距離當前路徑點太遠，則跳到下一個路徑點
        max_waypoint_distance = 5.0  # 你可以根據需要調整此閾值
        if distance_to_goal > max_waypoint_distance:
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(self.waypoints):
                self.done = True
                return self.state, 100, self.done, {}
            return self.state, 0, False, {}

        # 計算當前位置與上一個位置的移動距離
        if self.previous_robot_position is not None:
            distance_moved = np.linalg.norm([robot_x - self.previous_robot_position[0], robot_y - self.previous_robot_position[1]])
        else:
            distance_moved = 0  # 初次執行時設定為0

        # 判斷是否接近當前路徑點
        if distance_to_goal < REFERENCE_DISTANCE_TOLERANCE:
            self.no_progress_steps = 0  # 有進展則重置
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(self.waypoints):
                self.done = True
                return self.state, 100, self.done, {}
        else:
            # 如果移動距離非常小，則累積無進展計數器
            if distance_moved < 0.01:
                self.no_progress_steps += 1
                if self.no_progress_steps >= self.max_no_progress_steps:
                    rospy.loginfo("No progress detected, resetting environment.")
                    reward = -100.0  # 無進展懲罰
                    self.reset()
                    return self.state, reward, True, {}
            else:
                self.no_progress_steps = 0  # 有移動則重置計數器

        # 更新 previous_robot_position
        self.previous_robot_position = (robot_x, robot_y)

        # 使用純追蹤演算法來調整行動
        action = self.calculate_action_pure_pursuit()

        # 發送控制指令
        linear_speed = np.clip(action[0], -2.0, 2.0)
        steer_angle = np.clip(action[1], -0.6, 0.6)

        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = steer_angle
        self.pub_cmd_vel.publish(twist)

        # 發送 IMU 數據
        imu_data = self.generate_imu_data()
        self.pub_imu.publish(imu_data)

        rospy.sleep(0.1)

        reward, _ = self.calculate_reward(current_waypoint_x, current_waypoint_y)

        return self.state, reward, self.done, {}

    def reset(self):
        # Existing setup code
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

        # 優先使用優化過的路徑點，否則使用生成的原始路徑點
        if hasattr(self, 'optimized_waypoints') and self.optimized_waypoints:
            self.waypoints = self.optimized_waypoints
        else:
            self.waypoints = self.generate_waypoints()

        # Reset other attributes
        self.current_waypoint_index = 0
        self.done = False

        empty_lidar_data = PointCloud2()
        current_waypoint_x, current_waypoint_y = self.waypoints[self.current_waypoint_index]
        self.state = self.generate_occupancy_grid(empty_lidar_data, current_waypoint_x, current_waypoint_y)

        self.last_twist = Twist()
        self.pub_cmd_vel.publish(self.last_twist)

        imu_data = self.generate_imu_data()
        self.pub_imu.publish(imu_data)

        self.previous_yaw_error = 0
        self.no_progress_steps = 0
        self.previous_distance_to_goal = None
        self.collision_detected = False

        return self.state

    def calculate_reward(self, target_x, target_y):
        robot_x, robot_y, robot_yaw = self.get_robot_position()
        reward = 0
        done = False

        # 1. 根据方向误差计算奖励
        direction_to_target = np.arctan2(target_y - robot_y, target_x - robot_x)
        yaw_diff = np.abs(direction_to_target - robot_yaw)
        reward += max(0, 5 - yaw_diff * 5)  # 误差越小，奖励越高

        # 2. 距离目标的奖励
        distance_to_goal = np.sqrt((target_x - robot_x) ** 2 + (target_y - robot_y) ** 2)
        reward += (1.0 / (distance_to_goal + 1e-5)) * 10

        # 3. 安全性奖励，远离障碍物时奖励
        if not self.is_point_near_obstacle(robot_x, robot_y, threshold=0.5):
            reward += 50  # 如果距离障碍物足够远，增加奖励

        # 4. 盲区检测并施加惩罚
        blind_spot_distances = self.calculate_blind_spot_edge_distances()
        for distance in blind_spot_distances:
            if distance < 0.5:  # 根据车辆盲区设置距离
                reward -= 50  # 处于盲区内增加惩罚

        # 5. 是否到达最终目标点
        if np.linalg.norm([robot_x - self.target_x, robot_y - self.target_y]) < 0.5:
            reward += 1000
            done = True

        return reward, done

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

    def calculate_action_pure_pursuit(self):
        robot_x, robot_y, robot_yaw = self.get_robot_position()

        # 動態調整前視距離（lookahead distance）
        linear_speed = np.linalg.norm([self.last_twist.linear.x, self.last_twist.linear.y])
        lookahead_distance = 2.0 + 0.5 * linear_speed  # 根據速度調整前視距離

        # 找到距離當前最近的路徑點
        closest_index = self.find_closest_waypoint(robot_x, robot_y)
        target_index = closest_index

        # 根據前視距離選擇參考的路徑點
        cumulative_distance = 0.0
        for i in range(closest_index, len(self.waypoints)):
            wp_x, wp_y = self.waypoints[i]
            dist_to_wp = np.linalg.norm([wp_x - robot_x, wp_y - robot_y])
            cumulative_distance += dist_to_wp
            if cumulative_distance >= lookahead_distance:
                target_index = i
                break

        # 獲取前視點座標
        target_x, target_y = self.waypoints[target_index]

        # 計算前視點的方向
        direction_to_target = np.arctan2(target_y - robot_y, target_x - robot_x)
        yaw_error = direction_to_target - robot_yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

        # 根據角度誤差調整速度
        if np.abs(yaw_error) > 0.3:
            linear_speed = 0.5
        elif np.abs(yaw_error) > 0.1:
            linear_speed = 1.0
        else:
            linear_speed = 1.5

        # 使用PD控制器調整轉向角度
        kp, kd = self.adjust_control_params(linear_speed)
        previous_yaw_error = getattr(self, 'previous_yaw_error', 0)
        current_yaw_error_rate = yaw_error - previous_yaw_error
        steer_angle = kp * yaw_error + kd * current_yaw_error_rate
        steer_angle = np.clip(steer_angle, -0.6, 0.6)

        self.previous_yaw_error = yaw_error

        return np.array([linear_speed, steer_angle])

    def find_closest_waypoint(self, x, y):
        # 找到與當前位置最接近的路徑點
        min_distance = float('inf')
        closest_index = 0
        for i, (wp_x, wp_y) in enumerate(self.waypoints):
            dist = np.linalg.norm([wp_x - x, wp_y - y])
            if dist < min_distance:
                min_distance = dist
                closest_index = i
        return closest_index
    
    def adjust_control_params(self, linear_speed):
        if linear_speed <= 0.5:
            kp = 0.5
            kd = 0.2
        elif linear_speed <= 1.0:
            kp = 0.4
            kd = 0.3
        else:
            kp = 0.3
            kd = 0.4
        return kp, kd

class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(128)

        self._to_linear = self._get_conv_output_size(observation_space)

        self.fc1 = nn.Linear(self._to_linear, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 128)

        self.actor = nn.Linear(128, action_space)
        self.critic = nn.Linear(128, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_space))
        
        # 新增一個分支來輸出 yaw 值
        self.yaw_head = nn.Linear(128, 1)

    def _get_conv_output_size(self, shape):
        x = torch.zeros(1, *shape)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(1, -1)
        return x.size(1)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)

        action_mean = self.actor(x)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        value = self.critic(x)

        # 增加 yaw 值的輸出
        yaw = self.yaw_head(x)

        return action_mean, action_std, value, yaw

    def act(self, state):
        action_mean, action_std, _, yaw = self(state)  # 同時輸出 yaw
        action = action_mean + action_std * torch.randn_like(action_std)
        action = torch.tanh(action)
        max_action = torch.tensor([2.0, 0.6], device=action.device)
        min_action = torch.tensor([-2.0, -0.6], device=action.device)
        action = min_action + (action + 1) * (max_action - min_action) / 2

        # 輸出行為（action）和 yaw
        return action.detach(), yaw.detach()
    
    def evaluate(self, state, action):
        action_mean, action_std, value, yaw = self(state)  # 同時計算 yaw
        dist = torch.distributions.Normal(action_mean, action_std)
        action_log_probs = dist.log_prob(action).sum(dim=-1, keepdim=True)
        dist_entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return action_log_probs, value, dist_entropy, yaw

def ppo_update(ppo_epochs, env, model, optimizer, memory, scaler):
    for _ in range(ppo_epochs):
        state_batch, action_batch, reward_batch, done_batch, next_state_batch, indices, weights = memory.sample(BATCH_SIZE)
        
        adjusted_lr = LEARNING_RATE * (weights.mean().item() + 1e-3)
        for param_group in optimizer.param_groups:
            param_group['lr'] = adjusted_lr

        with torch.no_grad():
            old_log_probs, _, _, _ = model.evaluate(state_batch, action_batch)
        old_log_probs = old_log_probs.detach()

        for _ in range(PPO_EPOCHS):
            with torch.amp.autocast('cuda'):
                log_probs, state_values, dist_entropy, _ = model.evaluate(state_batch, action_batch)
                advantages = reward_batch + (1 - done_batch) * GAMMA * model(next_state_batch)[2].detach() - state_values

                ratio = (log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_PARAM, 1 + CLIP_PARAM) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(state_values, reward_batch + (1 - done_batch) * GAMMA * model(next_state_batch)[2].detach())
                entropy_loss = -0.02 * dist_entropy.mean()  # 添加熵正则项
                loss = actor_loss + 0.5 * critic_loss + entropy_loss

            scaler.scale(loss).backward()
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

    num_episodes = 1000000
    best_test_reward = -np.inf

    for e in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        total_reward = 0
        start_time = time.time()

        for time_step in range(1500):
            action, yaw = model.act(state)
            action_np = action.detach().cpu().numpy()
            yaw_np = yaw.detach().cpu().numpy()
            
            next_state, reward, done, _ = env.step(action_np)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)

            memory.add(state.cpu().numpy(), action_np, reward, done, next_state.cpu().numpy())
            state = next_state
            total_reward += reward
            elapsed_time = time.time() - start_time

            if done or elapsed_time > 240:
                if elapsed_time > 240:
                    reward -= 1000.0
                    print(f"Episode {e} failed at time step {time_step}: time exceeded 240 sec.")
                break

        # 每圈結束後進行 PPO 更新
        ppo_update(PPO_EPOCHS, env, model, optimizer, memory, scaler)
        memory.clear()

        print(f"Episode {e}, Total Reward: {total_reward}")

        # 在每圈結束後進行路徑優化
        env.generate_optimized_waypoints()

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

if __name__ == '__main__':
    main()