#!/usr/bin/env python3
import rospy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import Imu
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
import sensor_msgs.point_cloud2 as pc2
from scipy.special import comb
from collections import namedtuple
import tf
from tf.transformations import quaternion_from_euler
import time
from torch.amp import GradScaler
import yaml
from PIL import Image
from skimage.draw import line
from queue import PriorityQueue

# 超參數
REFERENCE_DISTANCE_TOLERANCE = 0.65
MEMORY_SIZE = 10000
BATCH_SIZE = 256
GAMMA = 0.99
LEARNING_RATE = 0.0003
PPO_EPOCHS = 5
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
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32)
        if isinstance(reward, np.ndarray):
            reward = torch.tensor(reward, dtype=torch.float32)
        if isinstance(done, np.ndarray):
            done = torch.tensor(done, dtype=torch.float32)
        if isinstance(next_state, np.ndarray):
            next_state = torch.tensor(next_state, dtype=torch.float32)
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

        # Ensure all priorities are valid for sampling
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        # Handle NaN in priorities
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

        # Ensure all states are 4D tensors
        # 在 memory.sample() 返回數據後統一維度
        states = torch.stack([s.view(1, 3, 64, 64) if s.dim() == 3 else s for s in states])
        next_states = torch.stack([ns.view(1, 3, 64, 64) if ns.dim() == 3 else ns for ns in next_states])

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
    def __init__(self, model):
        rospy.init_node('gazebo_rl_agent', anonymous=True)
        if model is None:
            raise ValueError("A valid model must be provided to GazeboEnv.")
        self.model = model
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.pub_imu = rospy.Publisher('/imu/data', Imu, queue_size=10)
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
        self.waypoint_distances = self.calculate_waypoint_distances()   # 計算一整圈機器任要奏的大致距離
        self.current_waypoint_index = 0
        self.last_twist = Twist()
        self.epsilon = 0.05
        self.collision_detected = False
        self.previous_robot_position = None  # 初始化 previous_robot_position 為 None
        self.previous_distance_to_goal = None  # 初始化 previous_distance_to_goal 為 None
        self.max_neighbors = 8  # 根據可能的鄰居數設置

        self.max_no_progress_steps = 10
        self.no_progress_steps = 0
        
        # 新增屬性，標記是否已計算過優化路徑
        self.optimized_waypoints_calculated = False
        self.optimized_waypoints = []  # 儲存優化後的路徑點

        self.waypoint_failures = {i: 0 for i in range(len(self.waypoints))}

        # 加载SLAM地圖
        self.load_slam_map('/home/chihsun/catkin_ws/src/my_robot_control/scripts/my_map0924.yaml')

        self.optimize_waypoints_with_a_star()
        
    def load_slam_map(self, yaml_path):
        # 讀取 YAML 檔案
        with open(yaml_path, 'r') as file:
            map_metadata = yaml.safe_load(file)
            self.map_origin = map_metadata['origin']  # 地圖原點
            self.map_resolution = map_metadata['resolution']  # 地圖解析度
            png_path = map_metadata['image'].replace(".pgm", ".png")  # 修改為png檔案路徑
            
            # 使用 PIL 讀取PNG檔
            png_image = Image.open(png_path).convert('L')
            self.slam_map = np.array(png_image)  # 轉為NumPy陣列


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
    
    def calculate_waypoint_distances(self):
        """
        計算每對相鄰 waypoint 之間的距離，並返回一個距離列表。
        """
        distances = []
        for i in range(len(self.waypoints) - 1):
            start_wp = self.waypoints[i]
            next_wp = self.waypoints[i + 1]
            distance = np.linalg.norm([next_wp[0] - start_wp[0], next_wp[1] - start_wp[1]])
            distances.append(distance)
        return distances


    def gazebo_to_image_coords(self, gazebo_x, gazebo_y):
        img_x = 2000 + gazebo_x * 20
        img_y = 2000 - gazebo_y * 20
        return int(img_x), int(img_y)

    def image_to_gazebo_coords(self, img_x, img_y):
        gazebo_x = (img_x - 2000) / 20
        gazebo_y = (2000 - img_y) / 20
        return gazebo_x, gazebo_y
    
    def heuristic(self, current, goal):
        """
        計算啟發式估計成本（比如歐幾里得距離）。
        """
        return np.linalg.norm(np.array(current) - np.array(goal))
    
    def get_neighbors(self, current, search_radius=25):
        """
        獲取當前節點周圍 `50x50` 的鄰居節點（以像素為單位）。
        """
        x, y = current
        neighbors = []
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx, ny = int(x + dx), int(y + dy)
                # 確保鄰居節點在地圖範圍內，且不是障礙物
                if 0 <= nx < self.slam_map.shape[1] and 0 <= ny < self.slam_map.shape[0]:
                    if self.slam_map[ny, nx] >= 250:  # 250 表示可通行
                        neighbors.append((nx, ny))
        return neighbors

    def a_star_with_rl_intervention(self, slam_map, start, goal):
        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while not open_set.empty():
            _, current = open_set.get()

            if current == goal:
                return self.reconstruct_path(came_from, current)

            # 提取節點的 50x50 局部地圖
            local_map = self.extract_local_map(current, size=50)

            # 將局部地圖和當前狀態輸入到 RL 模型
            state = self.generate_rl_state(local_map, current, f_score)
            rl_action, _ = self.model.act(state)
            rl_selected_neighbor = self.adjust_neighbor_from_action(current, rl_action)

            if rl_selected_neighbor not in neighbors:
                neighbors.append(rl_selected_neighbor)
            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                tentative_g_score = g_score[current] + self.cost(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    came_from[neighbor] = current
                    open_set.put((f_score[neighbor], neighbor))

        return None  # 如果無法到達目標
    
    def calculate_action_reward(self, current, rl_neighbor, a_star_neighbor, goal):
        """
        根據 RL 選擇的擴展點與 A* 原始邏輯的差異設計獎勵。
        """
        # 獎勵 RL 選擇的點是否更靠近目標
        rl_distance_to_goal = np.linalg.norm(np.array(goal) - np.array(rl_neighbor))
        a_star_distance_to_goal = np.linalg.norm(np.array(goal) - np.array(a_star_neighbor))

        # 如果 RL 選擇的更優，給予正獎勵，否則懲罰
        if rl_distance_to_goal < a_star_distance_to_goal:
            reward = 1.0  # 正獎勵
        else:
            reward = -1.0  # 懲罰

        return reward

    def is_line_free(self, png_image, start, end):
        """
        檢查從 start 到 end 的直線上是否有障礙物。
        使用 Bresenham 演算法來生成線上的點，並檢查這些點是否可通行。
        """
        rr, cc = line(start[1], start[0], end[1], end[0])  # 使用 skimage.draw.line 獲取直線上的點
        for r, c in zip(rr, cc):
            if not (0 <= r < png_image.shape[0] and 0 <= c < png_image.shape[1]):
                return False  # 如果點超出圖片範圍
            if png_image[r, c] < 250:  # 假設低於 250 為障礙物
                return False  # 直線上有障礙物
        return True
    
    def rl_intervention(self, current, neighbor, f_score):
        """
        強化學習模型決定新的擴展點。
        """
        state = self.generate_rl_state(current, neighbor, f_score)
        action = self.model.act(state)
        new_neighbor = self.adjust_neighbor(neighbor, action)
        return new_neighbor
        
    def generate_rl_state(self, local_map, current, f_score):
        """
        根據節點的局部地圖和路徑資訊生成狀態。
        """
        state = np.zeros((3, 50, 50), dtype=np.float32)  # 適配 50x50
        state[0, :, :] = local_map  # 障礙物分布
        state[1, :, :] = current[0]  # x 坐標信息
        state[2, :, :] = current[1]  # y 坐標信息

        return torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    
    def adjust_neighbor(self, neighbor, action):
        distance = np.linalg.norm(action[:2])
        angle = np.arctan2(action[1], action[0])
        dx = distance * np.cos(angle)
        dy = distance * np.sin(angle)
        return neighbor[0] + dx, neighbor[1] + dy
    
    def extract_local_map(self, current, size=50):
        """
        提取節點周圍 `50x50` 的局部地圖資訊。
        """
        img_x, img_y = self.gazebo_to_image_coords(*current)
        half_size = size // 2
        start_x = max(0, img_x - half_size)
        start_y = max(0, img_y - half_size)
        end_x = min(self.slam_map.shape[1], img_x + half_size)
        end_y = min(self.slam_map.shape[0], img_y + half_size)

        # 初始化局部地圖並填充數據
        local_map = np.zeros((size, size), dtype=np.float32)
        map_slice = self.slam_map[start_y:end_y, start_x:end_x]
        local_map[:map_slice.shape[0], :map_slice.shape[1]] = map_slice

        return local_map
    
    def pure_a_star(self, slam_map, start, goal):
        """
        單純的 A* 算法實現，無 RL 模型介入。
        """
        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while not open_set.empty():
            _, current = open_set.get()

            if current == goal:
                return self.reconstruct_path(came_from, current)

            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                # 檢查從 current 到 neighbor 是否有障礙物
                if not self.is_line_free(slam_map, current, neighbor):
                    continue

                tentative_g_score = g_score[current] + self.cost(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    open_set.put((f_score[neighbor], neighbor))

        return None  # 如果無法到達目標

    def optimize_waypoints_with_a_star(self):
        """
        使用純 A* 來優化路徑點，但僅在尚未計算過時執行。
        """
        if self.optimized_waypoints_calculated:
            rospy.loginfo("Using previously calculated optimized waypoints.")
            self.waypoints = self.optimized_waypoints  # 使用已計算的優化路徑
            return

        rospy.loginfo("Calculating optimized waypoints for the first time using pure A*.")
        optimized_waypoints = []

        # 地圖的有效範圍
        map_height, map_width = self.slam_map.shape

        for i in range(len(self.waypoints) - 1):
            start_point = (self.waypoints[i][0], self.waypoints[i][1])
            goal_point = (self.waypoints[i + 1][0], self.waypoints[i + 1][1])

            # 將起點和目標點轉換為地圖坐標
            img_start_x, img_start_y = self.gazebo_to_image_coords(*start_point)
            img_goal_x, img_goal_y = self.gazebo_to_image_coords(*goal_point)

            # 檢查邊界和可通行性
            if not (0 <= img_start_x < map_width and 0 <= img_start_y < map_height):
                rospy.logwarn(f"Start point {start_point} is out of bounds. Keeping original waypoint.")
                optimized_waypoints.append(start_point)
                continue

            if not (0 <= img_goal_x < map_width and 0 <= img_goal_y < map_height):
                rospy.logwarn(f"Goal point {goal_point} is out of bounds. Keeping original waypoint.")
                optimized_waypoints.append(goal_point)
                continue

            if self.slam_map[img_start_y, img_start_x] < 250:
                rospy.logwarn(f"Start point {start_point} is in an obstacle. Keeping original waypoint.")
                optimized_waypoints.append(start_point)
                continue

            if self.slam_map[img_goal_y, img_goal_x] < 250:
                rospy.logwarn(f"Goal point {goal_point} is in an obstacle. Keeping original waypoint.")
                optimized_waypoints.append(goal_point)
                continue

            # 調用純 A* 算法
            optimized_path = self.pure_a_star(self.slam_map, start_point, goal_point)
            if optimized_path:
                optimized_waypoints.extend(optimized_path)
            else:
                rospy.logwarn(f"Failed to optimize segment {i} ({start_point} -> {goal_point}). Keeping original waypoint.")
                optimized_waypoints.append(goal_point)  # 如果失敗，保持原路徑點

        self.optimized_waypoints = optimized_waypoints
        self.waypoints = optimized_waypoints
        self.optimized_waypoints_calculated = True

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

    def collision_callback(self, data):
        if len(data.states) > 0:
            self.collision_detected = True
            rospy.loginfo("Collision detected!")
        else:
            self.collision_detected = False

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

    def is_valid_data(self, data):
        for point in pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True):
            if point[0] != 0.0 or point[1] != 0.0 or point[2] != 0.0:
                return True
        return False

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

    def generate_occupancy_grid(self, robot_x, robot_y, grid_size=0.05, map_size=100):
        # 將機器人的座標轉換為地圖上的像素座標
        img_x, img_y = self.gazebo_to_image_coords(robot_x, robot_y)
        
        # 計算64x64網格在圖片上的起始和結束索引
        half_grid = 32  # 因為需要64x64的矩陣，所以邊長的一半是32
        start_x = max(0, img_x - half_grid)
        start_y = max(0, img_y - half_grid)
        end_x = min(self.slam_map.shape[1], img_x + half_grid)
        end_y = min(self.slam_map.shape[0], img_y + half_grid)

        # 擷取圖片中的64x64區域
        grid = np.zeros((64, 64), dtype=np.float32)
        grid_slice = self.slam_map[start_y:end_y, start_x:end_x]
        
        # 填充grid，將超出地圖範圍的部分填充為0
        grid[:grid_slice.shape[0], :grid_slice.shape[1]] = grid_slice

        # 將當前機器人位置資訊添加到occupancy grid
        occupancy_grid = np.zeros((3, 64, 64), dtype=np.float32)
        occupancy_grid[0, :, :] = grid
        occupancy_grid[1, :, :] = robot_x  # 機器人的x位置
        occupancy_grid[2, :, :] = robot_y  # 機器人的y位置

        return occupancy_grid

    def step(self, action):
        reward = 0
        robot_x, robot_y, robot_yaw = self.get_robot_position()
        self.state = self.generate_occupancy_grid(robot_x, robot_y)

        # 計算當前機器人位置與所有 waypoints 的距離，並找到距離最近的 waypoint 的索引
        distances = [np.linalg.norm([robot_x - wp_x, robot_y - wp_y]) for wp_x, wp_y in self.waypoints]
        closest_index = np.argmin(distances)
        if closest_index > self.current_waypoint_index:
            distance_reward = sum(self.waypoint_distances[self.current_waypoint_index:closest_index])
            reward += distance_reward * 100
            self.current_waypoint_index = closest_index
            print('distance to goal +', reward)

        # 確認最近的 waypoint 是否與目標位置相同
        current_waypoint_x, current_waypoint_y = self.waypoints[self.current_waypoint_index]

        # 計算距離移動量，並檢查是否有進展
        distance_moved = (
            np.linalg.norm([robot_x - self.previous_robot_position[0], robot_y - self.previous_robot_position[1]])
            if self.previous_robot_position
            else 0
        )
        self.previous_robot_position = (robot_x, robot_y)

        # 判斷是否需要 RL 介入 A* 算法
        if distance_moved < 0.03:
            self.no_progress_steps += 1
            if self.no_progress_steps >= self.max_no_progress_steps:
                self.waypoint_failures[self.current_waypoint_index] += 1
                print(f"Failure at point {self.current_waypoint_index}. RL intervention in A*")
                rospy.loginfo("No progress detected, re-planning path with RL.")

                # RL 介入 A* 算法，重新生成路徑
                new_path = self.a_star_with_rl_intervention((robot_x, robot_y), (self.target_x, self.target_y))
                if new_path:
                    print("New path generated by RL intervention.")
                    self.waypoints = new_path  # 更新路徑
                    self.waypoint_distances = self.calculate_waypoint_distances()
                    self.current_waypoint_index = 0
                    self.no_progress_steps = 0
                else:
                    print("RL intervention failed to generate a valid path.")
                    reward = -2000.0
                    self.reset()
                    return self.state, reward, True, {}

        else:
            self.no_progress_steps = 0

        # 根據路徑執行控制
        use_deep_rl_control = any(
            self.waypoint_failures.get(i, 0) > 1
            for i in range(max(0, self.current_waypoint_index - 3), min(len(self.waypoints), self.current_waypoint_index + 4))
        )

        if use_deep_rl_control:
            print('Operate by RL.')
            if isinstance(self.state, np.ndarray):
                self.state = torch.tensor(self.state, dtype=torch.float32).view(1, 3, 64, 64).to(device)
            elif self.state.dim() != 4:
                self.state = self.state.view(1, 3, 64, 64)

            action = self.model.act(self.state)
            print("RL Action output:", action)
            linear_speed = np.clip(action[0, 0].item(), -2.0, 2.0)
            steer_angle = np.clip(action[0, 1].item(), -0.6, 0.6)
        else:
            # 基於 A* 路徑的 Pure Pursuit 控制
            action = self.calculate_action_pure_pursuit()
            print("A* Action output:", action)
            linear_speed = np.clip(action[0], -2.0, 3.0)
            steer_angle = np.clip(action[1], -0.6, 0.6)

        # 發布控制指令
        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = steer_angle
        self.pub_cmd_vel.publish(twist)

        # 發布 IMU 資料
        imu_data = self.generate_imu_data()
        self.pub_imu.publish(imu_data)

        rospy.sleep(0.1)

        # 計算獎勵
        reward, _ = self.calculate_reward(robot_x, robot_y, reward, self.state)

        print(f"Reward: {reward}")
        return self.state, reward, self.done, {}

    def reset(self):

        robot_x, robot_y,_ = self.get_robot_position()
        self.state = self.generate_occupancy_grid(robot_x, robot_y)

        # 設置初始機器人位置和姿態
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

        # 確保使用優化過的路徑點
        if hasattr(self, 'waypoints') and self.waypoints:
            rospy.loginfo("Using optimized waypoints for reset.")
        else:
            rospy.loginfo("No optimized waypoints found, generating new waypoints.")
            self.waypoints = self.generate_waypoints()

        self.current_waypoint_index = 0
        self.done = False

        self.last_twist = Twist()
        self.pub_cmd_vel.publish(self.last_twist)

        imu_data = self.generate_imu_data()
        self.pub_imu.publish(imu_data)

        self.previous_yaw_error = 0
        self.no_progress_steps = 0
        self.previous_distance_to_goal = None
        self.collision_detected = False

        # Ensure the state is 4D tensor
        if isinstance(self.state, np.ndarray):
            self.state = torch.tensor(self.state, dtype=torch.float32).view(1, 3, 64, 64).to(device)
        elif self.state.dim() != 4:
            self.state = self.state.view(1, 3, 64, 64)

        return self.state

    def calculate_reward(self, robot_x, robot_y, reward, state):
        done = False

        # 確保 state 是 numpy 格式
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()

        # 確定佔據網格的數據格式
        if state.ndim == 4:
            occupancy_grid = state[0, 0]  # 第 0 批次的第一層
        elif state.ndim == 3:
            occupancy_grid = state[0]  # 直接取第一層
        else:
            raise ValueError("Unexpected state dimension. Expected 3D or 4D, got {}D.".format(state.ndim))

        # 機器人的當前圖像座標
        img_x, img_y = self.gazebo_to_image_coords(robot_x, robot_y)

        # 1. 獎勵：靠近目標的距離
        distance_to_goal = np.linalg.norm([robot_x - self.target_x, robot_y - self.target_y])
        if self.previous_distance_to_goal is not None:
            progress_reward = max(0, self.previous_distance_to_goal - distance_to_goal) * 100
            reward += progress_reward
            print(f"Progress reward: {progress_reward}")
        self.previous_distance_to_goal = distance_to_goal

        # 2. 懲罰：撞到障礙物
        obstacle_count = np.sum(occupancy_grid <= 190)  # 計算當前佔據網格中障礙物的數量
        obstacle_penalty = obstacle_count * 3
        reward -= obstacle_penalty
        print(f"Obstacle penalty: {obstacle_penalty}")

        # 3. 懲罰：過度震蕩或偏離路徑
        # 使用車輛的平滑性評估，例如前後車速或方向改變過大
        if self.previous_robot_position:
            movement_distance = np.linalg.norm(
                [robot_x - self.previous_robot_position[0], robot_y - self.previous_robot_position[1]]
            )
            if movement_distance < 0.05:  # 若移動過小，可能卡住
                reward -= 50
                print("Low movement penalty applied.")

        # 4. 獎勵：到達目標
        if distance_to_goal < 0.2:  # 假設 0.2 米以內為成功到達目標
            reward += 1000
            done = True
            print("Goal reached! Reward applied.")

        # 5. 超時處理（可以根據需求調整）
        if self.no_progress_steps >= self.max_no_progress_steps:
            reward -= 1000
            done = True
            print("Episode failed due to no progress.")

        self.previous_robot_position = (robot_x, robot_y)

        print(f"Total reward: {reward}")
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

        # 定義角度範圍，以當前車輛的yaw為中心
        angle_range = np.deg2rad(40)  # ±40度的範圍
        closest_index = None
        min_distance = float('inf')

        # 尋找該範圍內的最近路徑點
        for i in range(self.current_waypoint_index, len(self.waypoints)):
            wp_x, wp_y = self.waypoints[i]
            dist_to_wp = np.linalg.norm([wp_x - robot_x, wp_y - robot_y])
            direction_to_wp = np.arctan2(wp_y - robot_y, wp_x - robot_x)

            # 計算該點相對於當前車輛朝向的角度
            yaw_diff = direction_to_wp - robot_yaw
            yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))  # 確保角度在[-pi, pi]範圍內

            # 如果點位於yaw ± 35度範圍內，並且距離更近
            if np.abs(yaw_diff) < angle_range and dist_to_wp < min_distance:
                min_distance = dist_to_wp
                closest_index = i

        # 如果沒有找到符合條件的點，則繼續使用原始最近點
        if closest_index is None:
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
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))  # 確保角度在[-pi, pi]範圍內

        # 根據角度誤差調整速度
        if np.abs(yaw_error) > 0.3:
            linear_speed = 0.5
        elif np.abs(yaw_error) > 0.1:
            linear_speed = 1.0
        else:
            linear_speed = 2

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

        self.actor = nn.Linear(128, action_space)  # 對應的 action_space 是候選擴展點數量
        self.critic = nn.Linear(128, 1)

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

        action_logits = self.actor(x)  # 動作概率的 logits
        value = self.critic(x)

        return action_logits, value

    def act(self, state):
        if state.dim() < 4:
            state = state.unsqueeze(0)  # 添加一個 batch 維度
        elif state.dim() > 4:
            state = state.squeeze(0)  # 去掉多餘的維度
        action_logits, _ = self(state)
        action_probs = torch.softmax(action_logits, dim=-1)
        action = torch.multinomial(action_probs, 1)  # 采样动作
        return action, action_probs

    def evaluate(self, state, action):
        action_logits, value = self(state)
        action_probs = torch.softmax(action_logits, dim=-1)
        action_log_probs = torch.log(action_probs.gather(1, action.unsqueeze(-1)))  # 選擇已執行動作的 log prob
        dist_entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=-1)  # 熵
        return action_log_probs, value, dist_entropy

def ppo_update(ppo_epochs, env, model, optimizer, memory, scaler):
    for _ in range(ppo_epochs):
        state_batch, action_batch, reward_batch, done_batch, next_state_batch, indices, weights = memory.sample(BATCH_SIZE)

        # Calculate old log_probs
        with torch.no_grad():
            old_action_mean, old_action_std, _ = model(state_batch)
            dist = torch.distributions.Normal(old_action_mean, old_action_std)
            old_log_probs = dist.log_prob(action_batch).sum(dim=-1, keepdim=True)

        # Update PPO
        for _ in range(PPO_EPOCHS):
            with torch.amp.autocast('cuda'):
                action_mean, action_std, state_values = model(state_batch)

                # Calculate new log_probs
                dist = torch.distributions.Normal(action_mean, action_std)
                log_probs = dist.log_prob(action_batch).sum(dim=-1, keepdim=True)

                # Calculate advantages with GAE
                with torch.no_grad():
                    next_state_values = model(next_state_batch)[2]
                    delta = reward_batch + (1 - done_batch) * GAMMA * next_state_values - state_values
                    advantages = torch.zeros_like(delta)
                    advantage = 0.0
                    for t in reversed(range(len(delta))):
                        advantage = delta[t] + GAMMA * 0.95 * advantage * (1 - done_batch[t])
                        advantages[t] = advantage
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # Ensure target_values and state_values shape match
                    target_values = reward_batch + (1 - done_batch) * GAMMA * next_state_values
                    target_values = target_values.unsqueeze(-1) if target_values.dim() == 1 else target_values

                # PPO Loss
                ratio = (log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_PARAM, 1 + CLIP_PARAM) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(state_values, target_values)
                entropy_loss = -0.01 * dist.entropy().mean()
                loss = actor_loss + 0.5 * critic_loss + entropy_loss

            # Update model
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            optimizer.zero_grad()  # 清零梯度
            scaler.update()

def main():
    model = ActorCritic((3, 64, 64), 8).to(device)  # 假設輸出為 8 個候選點
    env = GazeboEnv(model)  # 傳入模型
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler('cuda')
    memory = PrioritizedMemory(MEMORY_SIZE)

    model_path = "/home/chihsun/catkin_ws/src/my_robot_control/scripts/saved_model_ppo.pth"
    best_model_path = "/home/chihsun/catkin_ws/src/my_robot_control/scripts/best_model.pth"

    # 加載已有模型（如果存在）
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded existing model.")
    else:
        print("Training new model.")

    num_episodes = 1000000
    best_test_reward = -np.inf

    for episode in range(num_episodes):
        state = env.reset()  # 重置環境
        total_reward = 0
        rl_interventions = 0  # 記錄 RL 介入次數
        successful_interventions = 0
        start_time = time.time()

        for step_count in range(150):  # 每個 episode 的最大步數
            # 使用強化學習模型決定擴展點（旁觀學習）
            action, _ = model.act(state)  # 解包动作和概率
            next_state, reward, done, info = env.step(action.cpu().numpy())

            # 記錄是否由 RL 干預
            if env.no_progress_steps >= env.max_no_progress_steps:
                rl_interventions += 1
                # 執行 RL 干預後的 A* 路徑重規劃
                intervention_path = env.a_star_with_rl_intervention(
                    (env.previous_robot_position[0], env.previous_robot_position[1]),
                    (env.target_x, env.target_y)
                )
                if intervention_path is not None:
                    successful_interventions += 1  # 記錄成功干預次數
            
            # 將數據添加到記憶體
            memory.add(state.cpu().numpy(), action.cpu().numpy(), reward, done, next_state.cpu().numpy())

            state = next_state
            total_reward += reward

            if done:
                break

        # 訓練 PPO 模型
        ppo_update(PPO_EPOCHS, env, model, optimizer, memory, scaler)
        memory.clear()

        # 日誌信息
        elapsed_time = time.time() - start_time
        print(
            f"Episode {episode}, Total Reward: {total_reward}, RL Interventions: {rl_interventions}, "
            f"Successful Interventions: {successful_interventions}, Elapsed Time: {elapsed_time:.2f}s"
        )

        # 保存最佳模型
        if total_reward > best_test_reward:
            best_test_reward = total_reward
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with reward: {best_test_reward}")

        # 定期保存模型
        if episode % 10 == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved after {episode} episodes.")

        rospy.sleep(1.0)  # 防止過快執行

    # 保存最終模型
    torch.save(model.state_dict(), model_path)
    print("Final model saved.")

if __name__ == '__main__':
    main()