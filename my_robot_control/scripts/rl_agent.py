#!/usr/bin/env python3
import rospy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState, ContactsState
from collections import namedtuple
import tf
from tf.transformations import quaternion_from_euler
import time
from torch.amp import GradScaler
import yaml
from PIL import Image
import random
import csv
import cv2
import datetime
from torch.optim.lr_scheduler import StepLR
from scipy.ndimage import distance_transform_edt

# 超參數
REFERENCE_DISTANCE_TOLERANCE = 0.65
MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.0003
PPO_EPOCHS = 5
CLIP_PARAM = 0.2
PREDICTION_HORIZON = 400
CONTROL_HORIZON = 10

device = torch.device("cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def grid_filter(obstacles, grid_size=0.5):
    obstacles = np.array(obstacles)
    # 按照 grid_size 取整
    grid_indices = (obstacles // grid_size).astype(int)
    # 找到唯一的网格
    unique_indices = np.unique(grid_indices, axis=0)
    # 返回网格中心点
    filtered_points = unique_indices * grid_size + grid_size / 2
    return filtered_points

class PrioritizedMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [None] * capacity
        self.position = 0
        self.priorities = torch.zeros((capacity,), dtype=torch.float32).cuda()
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.epsilon = 1e-6
        self.max_priority = 1.0

    def add(self, state, action, reward, done, next_state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        if isinstance(next_state, np.ndarray):
            next_state = torch.tensor(next_state, dtype=torch.float32)

        # Ensure states have the right shape (B, C, H, W)
        if state.dim() == 5:
            state = state.squeeze(1)
        if next_state.dim() == 5:
            next_state = next_state.squeeze(1)

        # Ensure states have 4 dimensions
        if state.dim() == 3:
            state = state.unsqueeze(0)
        if next_state.dim() == 3:
            next_state = next_state.unsqueeze(0)

        self.memory[self.position] = (
            state.to(device),
            torch.tensor(action, dtype=torch.float32, device=device),
            torch.tensor(reward, dtype=torch.float32, device=device),
            torch.tensor(done, dtype=torch.float32, device=device),
            next_state.to(device)
        )

    def sample(self, batch_size):
        valid_samples = [i for i, x in enumerate(self.memory) if x is not None]
        if len(valid_samples) < batch_size:
            raise ValueError(f"Not enough valid samples. Have {len(valid_samples)}, need {batch_size}")

        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(valid_samples)]
        probs = priorities ** self.alpha
        probs = probs / probs.sum()

        # Sample indices
        indices = torch.multinomial(probs, batch_size, replacement=False)
        
        # Calculate importance sampling weights
        weights = (len(valid_samples) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        # Get samples
        batch = [self.memory[idx] for idx in indices]
        states, actions, rewards, dones, next_states = zip(*batch)

        # Stack and transfer to device
        states = torch.stack(states).to(device)
        next_states = torch.stack(next_states).to(device)
        actions = torch.stack(actions).to(device)
        rewards = torch.stack(rewards).to(device)
        dones = torch.stack(dones).to(device)

        return states, actions, rewards, dones, next_states, indices, weights.to(device)


    def update_priorities(self, indices, priorities):
        priorities = priorities.abs().clamp(min=self.epsilon, max=100.0)
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max().item())

    def clear(self):
        self.position = 0
        self.memory = [None] * self.capacity
        self.priorities = torch.zeros((self.capacity,), dtype=torch.float32).cuda()

class GazeboEnv:
    def __init__(self, model):
        rospy.init_node('gazebo_rl_agent', anonymous=True)
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
        self.target_x = -7.2213
        self.target_y = -1.7003  
        self.waypoints = self.generate_waypoints()
        self.waypoint_distances = self.calculate_waypoint_distances()   # 計算一整圈機器任要奏的大致距離
        self.current_waypoint_index = 0
        self.last_twist = Twist()
        self.epsilon = 0.05
        self.collision_detected = False
        self.previous_robot_position = None  # 初始化 previous_robot_position 為 None
        self.previous_distance_to_goal = None  # 初始化 previous_distance_to_goal 為 None

        self.max_no_progress_steps = 10
        self.no_progress_steps = 0
        
        # 新增屬性，標記是否已計算過優化路徑
        self.optimized_waypoints_calculated = False
        self.optimized_waypoints = []  # 儲存優化後的路徑點

        self.waypoint_failures = {i: 0 for i in range(len(self.waypoints))}

        # 加载SLAM地圖
        self.load_slam_map('/home/chihsun/catkin_ws/src/my_robot_control/scripts/my_map0924.yaml')

        self.optimize_waypoints_with_a_star()
        
        self.sub_contacts = rospy.Subscriber('/gazebo/contacts', ContactsState, self.collision_callback)
    
    def load_slam_map(self, yaml_path):
        # 讀取 YAML 檔案
        with open(yaml_path, 'r') as file:
            map_metadata = yaml.safe_load(file)
            self.map_origin = map_metadata['origin']  # 地圖原點
            self.map_resolution = map_metadata['resolution']  # 地圖解析度
            png_path = map_metadata['image'].replace(".pgm", ".png")  # 修改為png檔案路徑
            
            # 使用 PIL 讀取PNG檔
            png_image = Image.open('/home/chihsun/catkin_ws/src/my_robot_control/scripts/my_map1203.png').convert('L')
            self.slam_map = np.array(png_image)  # 轉為NumPy陣列

        # 计算距离变换：障碍物为0，其它区域为1
        binary_map = (self.slam_map >= 250).astype(np.uint8)  # 可行区域设为1，障碍物设为0
        self.distance_transform = distance_transform_edt(binary_map) * self.map_resolution

    def generate_waypoints(self):
        waypoints = [(-6.4981, -1.0627),
            (-5.4541, -1.0117),
            (-4.4041, -0.862),
            (-3.3692, -1.0294),
            (-2.295, -1.114),
            (-1.2472, -1.0318),
            (-0.1614, -0.6948),
            (0.8931, -0.8804),
            (1.9412, -0.8604),
            (2.9804, -0.7229),
            (3.874, -0.2681),
            (4.9283, -0.1644),
            (5.9876, -0.345),
            (7.019, -0.5218),
            (7.9967, -0.2338),
            (9.0833, -0.1096),
            (10.1187, -0.3335),
            (11.1745, -0.6322),
            (12.1693, -0.8619),
            (13.1291, -0.4148),
            (14.1217, -0.0282),
            (15.1261, 0.123),
            (16.1313, 0.4439),
            (17.1389, 0.696),
            (18.1388, 0.6685),
            (19.2632, 0.5127),
            (20.2774, 0.2655),
            (21.2968, 0.0303),
            (22.3133, -0.0192),
            (23.2468, 0.446),
            (24.1412, 0.9065),
            (25.1178, 0.5027),
            (26.1279, 0.4794),
            (27.0867, 0.8266),
            (28.0713, 1.4229),
            (29.1537, 1.3866),
            (30.2492, 1.1549),
            (31.385, 1.0995),
            (32.4137, 1.243),
            (33.4134, 1.5432),
            (34.4137, 1.5904),
            (35.4936, 1.5904),
            (36.5067, 1.5607),
            (37.5432, 1.5505),
            (38.584, 1.7008),
            (39.6134, 1.9053),
            (40.5979, 2.0912),
            (41.6557, 2.3779),
            (42.5711, 2.8643),
            (43.5911, 2.9725),
            (44.5929, 3.0637),
            (45.5919, 2.9841),
            (46.6219, 2.9569),
            (47.6314, 3.0027),
            (48.7359, 2.832),
            (49.5462, 2.1761),
            (50.5982, 2.1709),
            (51.616, 2.3573),
            (52.6663, 2.5593),
            (53.7532, 2.5325),
            (54.7851, 2.5474),
            (55.8182, 2.5174),
            (56.8358, 2.6713),
            (57.8557, 2.8815),
            (58.8912, 3.0949),
            (59.7436, 3.6285),
            (60.5865, 4.2367),
            (60.6504, 5.2876),
            (60.7991, 6.3874),
            (60.322, 7.3094),
            (59.8004, 8.1976),
            (59.4093, 9.195),
            (59.1417, 10.1994),
            (59.1449, 11.2274),
            (59.5323, 12.2182),
            (59.8637, 13.2405),
            (60.5688, 14.0568),
            (60.6266, 15.1571),
            (60.007, 15.9558),
            (59.0539, 17.0128),
            (57.9671, 17.326),
            (56.9161, 16.7399),
            (55.9553, 17.0346),
            (54.9404, 17.0596),
            (53.9559, 16.8278),
            (52.9408, 16.8697),
            (51.9147, 16.7642),
            (50.9449, 16.4902),
            (49.9175, 16.3029),
            (48.8903, 16.1165),
            (47.7762, 16.0994),
            (46.7442, 16.0733),
            (45.7566, 15.8195),
            (44.756, 15.7218),
            (43.7254, 15.9309),
            (42.6292, 15.8439),
            (41.6163, 15.8177),
            (40.5832, 15.7881),
            (39.5617, 15.773),
            (38.5099, 15.5648),
            (37.692, 14.9481),
            (36.8538, 14.3078),
            (35.8906, 13.8384),
            (34.8551, 13.6316),
            (33.8205, 13.5495),
            (32.7391, 13.4423),
            (31.7035, 13.1056),
            (30.6971, 12.7802),
            (29.6914, 12.5216),
            (28.7072, 12.3238),
            (27.6442, 12.0953),
            (26.5991, 11.9873),
            (25.5713, 11.9867),
            (24.488, 12.0679),
            (23.4441, 12.0246),
            (22.3169, 11.7745),
            (21.3221, 11.538),
            (20.3265, 11.4243),
            (19.2855, 11.5028),
            (18.2164, 11.5491),
            (17.1238, 11.6235),
            (16.0574, 11.4029),
            (14.982, 11.2479),
            (13.9491, 11.0487),
            (12.9017, 11.1455),
            (11.8915, 11.4186),
            (10.8461, 11.6079),
            (9.9029, 12.0097),
            (9.0549, 12.5765),
            (8.4289, 13.4238),
            (7.4035, 13.6627),
            (6.3785, 13.5659),
            (5.3735, 13.4815),
            (4.3971, 13.1044),
            (3.3853, 13.2918),
            (2.3331, 13.0208),
            (1.2304, 12.9829),
            (0.2242, 13.094),
            (-0.807, 12.9358),
            (-1.8081, 12.8495),
            (-2.7738, 13.3168),
            (-3.4822, 14.0699),
            (-4.5285, 14.2483),
            (-5.5965, 13.9753),
            (-6.5324, 13.6016),
            (-7.3092, 12.8632),
            (-8.3255, 12.9916),
            (-9.1914, 13.7593),
            (-10.2374, 14.069),
            (-11.2162, 13.7566),
            (-11.653, 12.8061),
            (-11.6989, 11.7238),
            (-11.8899, 10.7353),
            (-12.6174, 10.0373),
            (-12.7701, 8.9551),
            (-12.4859, 7.9523),
            (-12.153, 6.8903),
            (-12.4712, 5.819),
            (-13.0498, 4.8729),
            (-13.1676, 3.8605),
            (-12.4328, 3.1822),
            (-12.1159, 2.1018),
            (-12.8436, 1.2659),
            (-13.3701, 0.2175),
            (-13.0514, -0.8866),
            (-12.3046, -1.619),
            (-11.2799, -1.472),
            (-10.1229, -1.3051),
            (-9.1283, -1.4767),
            (-8.1332, -1.2563),
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
        gazebo_x = (img_x - 2000) / 20.0
        gazebo_y = (2000 - img_y) / 20.0
        return gazebo_x, gazebo_y

    def calculate_heuristic(self, point, goal_point, png_image, cost_map, grid_size=50, weights=None):
        """
        修改启发式函数，将路径点作为软约束。
        """
        current = np.array(current, dtype=np.float64)

        # 动态计算候选目标点中最佳的点
        best_goal = None
        lowest_cost = float('inf')

        for goal in goal_candidates:
            goal = np.array(goal, dtype=np.float64)

            # 与目标的距离代价
            dist_to_goal = np.linalg.norm(goal - current)

            # 平滑性代价
            smoothness_penalty = 0.0
            if previous_point is not None:
                prev_point = np.array(previous_point, dtype=np.float64)
                current_to_goal_vec = goal - current
                prev_to_current_vec = current - prev_point

                cos_theta = np.dot(current_to_goal_vec, prev_to_current_vec) / (
                    np.linalg.norm(current_to_goal_vec) * np.linalg.norm(prev_to_current_vec) + 1e-5
                )
                angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                smoothness_penalty = smoothness_weight * (angle ** 2)

            # 障碍物代价
            obstacle_count = self.calculate_obstacle_count(goal)
            obstacle_penalty = obstacle_weight * obstacle_count

            # 距离路径点的偏离代价
            waypoint_penalty = 0.0
            if waypoint is not None:
                waypoint = np.array(waypoint, dtype=np.float64)
                dist_to_waypoint = np.linalg.norm(goal - waypoint)
                waypoint_penalty = waypoint_weight * dist_to_waypoint

            # 总成本
            total_cost = dist_to_goal + smoothness_penalty + obstacle_penalty + waypoint_penalty
            if total_cost < lowest_cost:
                lowest_cost = total_cost
                best_goal = goal

        return lowest_cost, best_goal
    
    def find_best_point_in_grid(self, waypoint, previous_waypoint=None, grid_size=100, waypoint_weight=1.0):
        img_x, img_y = self.gazebo_to_image_coords(*waypoint)
        half_grid = grid_size // 2

        # 限定搜索范围
        min_x = max(0, img_x - half_grid)
        max_x = min(self.slam_map.shape[1] - 1, img_x + half_grid)
        min_y = max(0, img_y - half_grid)
        max_y = min(self.slam_map.shape[0] - 1, img_y + half_grid)

        grid = self.slam_map[min_y:max_y+1, min_x:max_x+1]
        free_space = np.argwhere(grid >= 250)

        if free_space.size == 0:
            rospy.logwarn(f"No free space found around waypoint {waypoint}. Using original point.")
            return waypoint

        global_free_space = free_space + np.array([min_y, min_x])

        # 获取候选点的全局坐标
        goal_candidates = [self.image_to_gazebo_coords(x[1], x[0]) for x in global_free_space]

        # 计算最佳的目标点
        cost, best_point = self.heuristic_cost(
            (img_x, img_y),
            goal_candidates,
            previous_point=self.gazebo_to_image_coords(*previous_waypoint) if previous_waypoint is not None else None,
            waypoint=waypoint,
            waypoint_weight=waypoint_weight
        )

        if best_point is not None:
            return best_point
        else:
            rospy.logwarn(f"No optimal point found around waypoint {waypoint}. Using original point.")
            return waypoint

    def calculate_obstacle_count(self, point):
        """
        计算给定点周围障碍物的数量。
        """
        x, y = map(float, point)  # 确保输入点是浮点数
        search_range = 20  # 搜索范围（像素）

        # 限制搜索范围在地图边界内
        min_x = max(0, int(x - search_range))
        max_x = min(self.slam_map.shape[1], int(x + search_range))
        min_y = max(0, int(y - search_range))
        max_y = min(self.slam_map.shape[0], int(y + search_range))

        # 找到障碍物点
        obstacle_count = np.sum(self.slam_map[min_y:max_y, min_x:max_x] < 250)  # 障碍物像素值小于 250

        return obstacle_count  # 返回障碍物点的数量
    
    def is_line_free(self, png_image, current, neighbor, safe_threshold=230):
        current = np.array(current, dtype=np.float64)
        neighbor = np.array(neighbor, dtype=np.float64)
        
        # 使用 Bresenham 演算法生成 current 到 neighbor 的線段
        rr, cc = line(int(round(current[1])), int(round(current[0])),
                    int(round(neighbor[1])), int(round(neighbor[0])))

        # 如果未找到路徑，返回初始目標點
        rospy.logwarn("A* failed to find a path.")
        return goal_point

    def optimize_waypoints_with_a_star(self):
        """
        使用擴展的 A* 啟發函數優化路徑點。
        """
        if self.optimized_waypoints_calculated:
            rospy.loginfo("Using previously calculated optimized waypoints.")
            self.waypoints = self.optimized_waypoints
            return

        rospy.loginfo("Calculating optimized waypoints for the first time using extended A*.")
        optimized_waypoints = []
        total_waypoints = len(self.waypoints) - 1  # 總共需要優化的段數

        for i in range(total_waypoints):
            start_point = (self.waypoints[i][0], self.waypoints[i][1])
            goal_point = (self.waypoints[i + 1][0], self.waypoints[i + 1][1])

            # 輸出當前進度
            rospy.loginfo(f"Optimizing segment {i + 1}/{total_waypoints}: Start {start_point} -> Goal {goal_point}")

            # 優化當前段
            optimized_point = self.a_star_optimize_waypoint(self.slam_map, self.cost_map, start_point, goal_point)
            optimized_waypoints.append(optimized_point)

        optimized_waypoints.append(self.waypoints[-1])
        self.optimized_waypoints = optimized_waypoints
        self.waypoints = optimized_waypoints
        self.optimized_waypoints_calculated = True

        # 最終優化路徑可視化
        save_path = '/home/chihsun/catkin_ws/src/my_robot_control/scripts/optimized_path.png'
        self.visualize_complete_path(self.optimized_waypoints, save_path=save_path)
        rospy.loginfo(f"Global path optimization complete. Visualization saved to {save_path}.")
    
    def visualize_original_path(self, save_path='/home/chihsun/catkin_ws/src/my_robot_control/scripts/original_path.png'):
        """
        可视化原始的参考路径点，并保存为图片。
        """
        if not hasattr(self, 'slam_map'):
            raise ValueError("SLAM map not loaded.")

        # 转换地图为灰度图
        map_img = self.slam_map.copy()
        map_img[map_img < 250] = 0  # 障碍物区域
        map_img[map_img >= 250] = 255  # 可通行区域

        # 转换路径点到图像坐标
        img_points = [self.gazebo_to_image_coords(p[0], p[1]) for p in self.generate_waypoints()]  # 使用原始路径点

        # 绘制地图
        plt.figure(figsize=(10, 10))
        plt.imshow(map_img, cmap='gray', origin='upper')

        # 绘制路径点为单独的点
        for point in img_points:
            if 0 <= point[0] < map_img.shape[1] and 0 <= point[1] < map_img.shape[0]:
                plt.scatter(point[0], point[1], color='orange', s=5)  # 单独的点，大小为5

        # 标注起点和终点
        img_start = self.gazebo_to_image_coords(*self.generate_waypoints()[0])
        img_goal = self.gazebo_to_image_coords(*self.generate_waypoints()[-1])
        plt.scatter(img_start[0], img_start[1], color='red', label='Start', s=50)
        plt.scatter(img_goal[0], img_goal[1], color='blue', label='Goal', s=50)

        # 设置绘图范围
        plt.xlim(0, map_img.shape[1])
        plt.ylim(map_img.shape[0], 0)  # 注意：图像坐标 y 轴是倒置的

        # 添加图例并保存图片
        plt.legend()
        plt.title('Original Path Points Visualization')
        plt.savefig(save_path)
        plt.close()
        rospy.loginfo(f"Original path visualization saved to {save_path}")
    
    def visualize_complete_path(self, waypoints, save_path = f'/home/chihsun/catkin_ws/src/my_robot_control/scripts/full_path_{time.time()}.png'):
        """
        可视化路径点和当前位置，并在 costmap 上显示
        """
        if not hasattr(self, 'slam_map') or not hasattr(self, 'cost_map'):
            raise ValueError("SLAM map or cost map not loaded.")

        # 創建一個 RGB 圖像來顯示 cost map
        cost_map_rgb = np.zeros((self.cost_map.shape[0], self.cost_map.shape[1], 3), dtype=np.uint8)
        
        # 將不同代價值映射到不同顏色
        cost_map_rgb[self.cost_map == 0] = [255, 255, 255]      # 空白區域為白色
        cost_map_rgb[self.cost_map == 100] = [200, 200, 255]    # 外層膨脹區為淺藍色
        cost_map_rgb[self.cost_map == 190] = [150, 150, 255]    # 內層膨脹區為中藍色
        cost_map_rgb[self.cost_map == 254] = [100, 100, 100]    # 障礙物為灰色

        # 獲取當前機器人位置
        robot_x, robot_y, _ = self.get_robot_position()
        robot_img_x, robot_img_y = self.gazebo_to_image_coords(robot_x, robot_y)

        # 轉換路徑點到圖像坐標
        img_points = [self.gazebo_to_image_coords(p[0], p[1]) for p in waypoints]

        # 創建圖像
        plt.figure(figsize=(12, 12))
        plt.imshow(cost_map_rgb)

        # 繪製所有路徑點
        for i, point in enumerate(img_points):
            if 0 <= point[0] < cost_map_rgb.shape[1] and 0 <= point[1] < cost_map_rgb.shape[0]:
                if i == self.current_waypoint_index:
                    # 當前目標點用黃色標記
                    plt.scatter(point[0], point[1], color='yellow', s=100, marker='*', label='Current Target')
                else:
                    # 其他路徑點用綠色標記
                    plt.scatter(point[0], point[1], color='green', s=20)

        # 標記起點和終點
        img_start = self.gazebo_to_image_coords(*waypoints[0])
        img_goal = self.gazebo_to_image_coords(*waypoints[-1])
        plt.scatter(img_start[0], img_start[1], color='blue', s=100, marker='^', label='Start')
        plt.scatter(img_goal[0], img_goal[1], color='red', s=100, marker='v', label='Goal')

        # 標記當前機器人位置
        plt.scatter(robot_img_x, robot_img_y, color='purple', s=150, marker='o', label='Robot')

        # 添加圖例和標題
        plt.legend(fontsize=12)
        plt.title('Path Visualization with Cost Map', fontsize=14)
        
        # 設置軸的範圍
        plt.xlim(0, cost_map_rgb.shape[1])
        plt.ylim(cost_map_rgb.shape[0], 0)  # 注意：圖像坐標 y 軸是倒置的

        # 保存圖片
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        rospy.loginfo(f"Path visualization saved to {save_path}")

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

    def generate_occupancy_grid(self, robot_x, robot_y, linear_speed, steer_angle, grid_size=0.05, map_size=100):
        # 将机器人的坐标转换为地图上的像素坐标

        linear_speed = np.clip(linear_speed, -2.0, 2.0)
        steer_angle = np.clip(steer_angle, -0.5, 0.5)

        img_x, img_y = self.gazebo_to_image_coords(robot_x, robot_y)

        # 计算64x64网格在图片上的起始和结束索引
        half_grid = 32
        start_x = max(0, img_x - half_grid)
        start_y = max(0, img_y - half_grid)
        end_x = min(self.slam_map.shape[1], img_x + half_grid)
        end_y = min(self.slam_map.shape[0], img_y + half_grid)

        # 提取图片中的64x64区域
        grid = np.zeros((64, 64), dtype=np.float32)
        grid_slice = self.slam_map[start_y:end_y, start_x:end_x]

        # 填充 grid，将超出地图范围的部分填充为0
        grid[:grid_slice.shape[0], :grid_slice.shape[1]] = grid_slice

        # 将当前机器人位置信息添加到占据栅格
        occupancy_grid = np.zeros((3, 64, 64), dtype=np.float32)
        
        # 第一层：归一化图片数据到 [0, 1]
        occupancy_grid[0, :, :] = grid/255.0

        # 第二层：归一化速度到 [0, 1]
        occupancy_grid[1, :, :] = (linear_speed + 2.0)/4

        # 第三层：归一化角度到 [0, 1]
        occupancy_grid[2, :, :] = (steer_angle + 0.5)/1.0

        if np.isnan(occupancy_grid).any() or np.isinf(occupancy_grid).any():
            raise ValueError("NaN or Inf detected in occupancy_grid!")
        return occupancy_grid

    def step(self, action, obstacles):
        reward = 0
        robot_x, robot_y, robot_yaw = self.get_robot_position()

        # 确保 action 是一维数组
        action = np.squeeze(action)
        linear_speed = np.clip(action[0], -2.0, 2.0)
        steer_angle = np.clip(action[1], -0.5, 0.5)
        print("linear speed = ", linear_speed, " steer angle = ", steer_angle)

        # 更新状态
        self.state = self.generate_occupancy_grid(robot_x, robot_y, linear_speed, steer_angle)

        distances = [np.linalg.norm([robot_x - wp_x, robot_y - wp_y]) for wp_x, wp_y in self.waypoints]
        closest_index = np.argmin(distances)

        if closest_index > self.current_waypoint_index:
            self.current_waypoint_index = closest_index

        distance_to_goal = np.linalg.norm([robot_x - self.target_x, robot_y - self.target_y])

        if distance_to_goal < 0.5:  # 设定阈值为0.5米，可根据需要调整
            print('Robot has reached the goal!')
            reward += 20.0 # 给一个大的正向奖励
            self.reset()
            return self.state, reward, True, {}  # 重置环境

        if self.current_waypoint_index < len(self.waypoints):
            current_wp = self.waypoints[self.current_waypoint_index]
            distance_to_wp = np.linalg.norm([robot_x - current_wp[0], robot_y - current_wp[1]])
            if distance_to_wp < 0.5:  # 假設通過 waypoint 的距離閾值為 0.5
                reward += 1.0  # 通過 waypoint 獎勵

        # 更新机器人位置
        if self.previous_robot_position is not None:
            distance_moved = np.linalg.norm([
                robot_x - self.previous_robot_position[0],
                robot_y - self.previous_robot_position[1]
            ])
            reward += distance_moved*5  # 根据移动距离奖励
            # print("reward by distance_moved +", distance_moved)
        else:
            distance_moved = 0

        self.previous_robot_position = (robot_x, robot_y)

        # 检查是否需要使用 RL 控制
        failure_range = range(
            max(0, self.current_waypoint_index - 6),
            min(len(self.waypoints), self.current_waypoint_index + 2)
        )
        use_deep_rl_control = any(
            self.waypoint_failures.get(i, 0) > 1 for i in failure_range
        )

        collision = detect_collision(robot_x, robot_y, robot_yaw, obstacles)
        if not use_deep_rl_control:
            if collision:
                self.waypoint_failures[self.current_waypoint_index] += 1
                print('touch the obstacles')
                reward -= 10.0
                self.reset()
                return self.state, reward, True, {}
        
        # 处理无进展的情况
        if distance_moved < 0.05:
            self.no_progress_steps += 1
            reward -= 0.3
            if self.no_progress_steps >= self.max_no_progress_steps:
                if use_deep_rl_control:
                    print('failure at point', self.current_waypoint_index)
                    rospy.loginfo("No progress detected, resetting environment.")
                    reward -= 10.0
                    self.reset()
                    return self.state, reward, True, {}
                else:
                    self.waypoint_failures[self.current_waypoint_index] += 1
                    print('failure at point', self.current_waypoint_index)
                    rospy.loginfo("No progress detected, resetting environment.")
                    reward -= 10.0
                    self.reset()
                    return self.state, reward, True, {}
        else:
            self.no_progress_steps = 0
        
        if self.collision_detected:
            rospy.loginfo('collision detectd! resetting')
            reward -= 10.0
            self.reset()
            return self.state, reward, True, {}
        # 发布控制命令
        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = steer_angle
        self.pub_cmd_vel.publish(twist)
        self.last_twist = twist

        imu_data = self.generate_imu_data()
        self.pub_imu.publish(imu_data)

        rospy.sleep(0.1)

        if isinstance(self.state, np.ndarray):
            self.state = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0).to(device)  # 增加 batch 维度
        elif self.state.dim() != 4:
            self.state = self.state.unsqueeze(0)  # 增加 batch 维度
        reward, _ = self.calculate_reward(robot_x, robot_y, reward, self.state)
        print('reward = ',reward)
        return self.state, reward, self.done, {}

    def reset(self):

        robot_x, robot_y,_ = self.get_robot_position()
        self.state = self.generate_occupancy_grid(robot_x, robot_y, linear_speed=0, steer_angle=0)

        # 設置初始機器人位置和姿態
        yaw = -0.0053
        quaternion = quaternion_from_euler(0.0, 0.0, yaw)
        state_msg = ModelState()
        state_msg.model_name = 'my_robot'
        state_msg.pose.position.x = -6.4981
        state_msg.pose.position.y = -1.0627
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
            self.state = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0).to(device)
        elif self.state.dim() != 4:
            self.state = self.state.unsqueeze(0)
        return self.state


    def calculate_reward(self, robot_x, robot_y, reward, state):
        done = False
        # 將機器人的座標轉換為地圖上的坐標
        
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if state.ndim == 4:
            # 对于 4 维情况，取第一个批次数据中的第一层
            occupancy_grid = state[0, 0]
        elif state.ndim == 3:
            # 对于 3 维情况，直接取第一层
            occupancy_grid = state[0]

        img_x, img_y = self.gazebo_to_image_coords(robot_x, robot_y)
        obstacle_count = np.sum(occupancy_grid <= 190/255.0)  # 假設state[0]為佔據網格通道
        reward += 5 - obstacle_count*3/100.0

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
        lookahead_distance = 1.2 + 0.5 * linear_speed  # 根據速度調整前視距離

        # 定義角度範圍，以當前車輛的yaw為中心
        angle_range = np.deg2rad(30)  # ±40度的範圍
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
            linear_speed = 3

        # 使用PD控制器調整轉向角度
        kp, kd = self.adjust_control_params(linear_speed)
        previous_yaw_error = getattr(self, 'previous_yaw_error', 0)
        current_yaw_error_rate = yaw_error - previous_yaw_error
        steer_angle = kp * yaw_error + kd * current_yaw_error_rate
        steer_angle = np.clip(steer_angle, -0.5, 0.5)

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
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(self._get_conv_output_size(observation_space), 256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)

        self.actor_mean = nn.Linear(128, action_space)
        self.actor_std = nn.Linear(128, action_space)  # 动态方差
        self.critic = nn.Linear(128, 1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        

    def _get_conv_output_size(self, shape):
        x = torch.zeros(1, *shape)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = x.view(1, -1)
        return x.size(1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))

        x, _ = self.lstm(x.unsqueeze(0))  # 时间序列处理
        x = x.squeeze(0)

        action_mean = self.actor_mean(x)
        action_std = torch.clamp(torch.exp(self.actor_std(x)), min=1e-3, max=2.0)  # 限制标准差范围
        value = self.critic(x)

        return action_mean, action_std, value


    def act(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32).to(device)
        
        # Ensure proper dimensions
        while state.dim() > 4:
            state = state.squeeze(0)
        while state.dim() < 4:
            state = state.unsqueeze(0)
        
        action_mean, action_std, _ = self(state)
        
        # Define action bounds
        action_space = torch.tensor([
            [0.5, -0.5],  # min values
            [2.0, 0.5]    # max values
        ], device=device)
        
        # Scale actions to proper range
        scaled_mean = action_space[0] + (torch.tanh(action_mean) + 1.0) * (action_space[1] - action_space[0]) / 2.0
        
        if self.training:
            # Add bounded noise during training
            noise = torch.clamp(torch.randn_like(action_std) * action_std, -0.5, 0.5)
            action = torch.clamp(scaled_mean + noise, action_space[0], action_space[1])

        else:
            action = scaled_mean
                
        return action.detach()

    def evaluate(self, state, action):
        action_mean, action_std, value = self(state)
        
        # Ensure proper bounds for action_std
        action_std = torch.clamp(action_std, min=1e-6, max=1.0)
        
        # Create normal distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        
        # Calculate log probabilities
        action_log_probs = dist.log_prob(action).sum(dim=-1, keepdim=True)
        dist_entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return action_log_probs, value, dist_entropy

class DWA:
    def __init__(self, goal):
        self.max_speed = 2
        self.max_yaw_rate = 0.5
        self.dt = 0.2
        self.predict_time = 3.0
        self.goal = goal
        self.robot_radius = 0.3

    def calc_dynamic_window(self, state):
        # 當前速度限制
        vs = [0, self.max_speed, -self.max_yaw_rate, self.max_yaw_rate]
        dw = vs
        return dw

    def motion(self, state, control):
        # 運動模型計算下一步
        x, y, theta, v, omega = state
        next_x = x + v * np.cos(theta) * self.dt
        next_y = y + v * np.sin(theta) * self.dt
        next_theta = theta + omega * self.dt
        next_v = control[0]
        next_omega = control[1]
        return [next_x, next_y, next_theta, next_v, next_omega]

    def calc_trajectory(self, state, control):
        # 預測軌跡
        trajectory = [state]
        for _ in range(int(self.predict_time / self.dt)):
            state = self.motion(state, control)
            trajectory.append(state)
        return np.array(trajectory)

    def calc_score(self, trajectory, obstacles):
        # 目标距离分数
        x, y = trajectory[-1, 0], trajectory[-1, 1]
        goal_dist = np.sqrt((self.goal[0] - x) ** 2 + (self.goal[1] - y) ** 2)
        goal_score = -goal_dist

        # 安全分数：检测轨迹中是否发生碰撞
        clearance_score = float('inf')
        for tx, ty, _, _, _ in trajectory:
            for ox, oy in obstacles:
                dist = np.sqrt((ox - tx) ** 2 + (oy - ty) ** 2)
                if dist < self.robot_radius:
                    return goal_score, -100.0, 0.0  # 如果发生碰撞，直接返回最低分
                clearance_score = min(clearance_score, dist)

        # 速度分数
        speed_score = trajectory[-1, 3]  # 最终速度
        return goal_score, clearance_score, speed_score

    def plan(self, state, obstacles):
        print("dwa goal: ", self.goal)
        # 獲取動態窗口
        obstacles = [(ox, oy) for ox, oy in obstacles]
        dw = self.calc_dynamic_window(state)  # 速度 角度限制
        # 遍歷動態窗口中的所有控制
        best_trajectory = None
        best_score = -100.0
        best_control = [0.0, 0.0]
        # print("Dynamic Window", dw)
        for v in np.arange(dw[0], dw[1], 0.2):  # 線速度範圍
            for omega in np.arange(dw[2], dw[3], 0.1):  # 角速度範圍

                # 模擬軌跡
                control = [v, omega]
                trajectory = self.calc_trajectory(state, control)
                # 計算評分函數
                goal_score, clearance_score, speed_score = self.calc_score(trajectory, obstacles)
                total_score = goal_score * 0.5 + clearance_score * 0.45  + speed_score * 0.05

                # 找到最佳控制
                if total_score > best_score:
                    best_score = total_score
                    best_trajectory = trajectory
                    best_control = control
        print(f"v: {v}, omega: {omega}, goal_score: {goal_score}, clearance_score: {clearance_score}, speed_score: {speed_score}")
        return best_control, best_trajectory

def ppo_update(ppo_epochs, env, model, optimizer, memory, scaler, batch_size, scheduler):
    valid_samples = len([x for x in memory.memory if x is not None])
    if valid_samples < batch_size:
        return

    accumulation_steps = 4
    warmup_epochs = 10
    optimizer.zero_grad()

    for epoch in range(ppo_epochs):
        # Warmup learning rate
        progress = min(1.0, epoch / warmup_epochs)
        adjusted_lr = LEARNING_RATE * progress * (1 / (1 + epoch * 0.001))
        for param_group in optimizer.param_groups:
            param_group['lr'] = adjusted_lr

        # Sample batch
        state_batch, action_batch, reward_batch, done_batch, next_state_batch, indices, weights = memory.sample(batch_size)
        state_batch, next_state_batch = _adjust_dimensions(state_batch, next_state_batch)
        
        # Normalize rewards
        reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-8)
        reward_batch = torch.clamp(reward_batch, -10.0, 10.0)

        # Calculate advantages
        with torch.no_grad():
            old_log_probs, _, _ = model.evaluate(state_batch, action_batch)
            _, _, next_values = model(next_state_batch)
            _, _, values = model(state_batch)

        # Compute target values and advantages
        target_values = reward_batch + (1 - done_batch) * GAMMA * next_values
        advantages = (target_values - values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.clamp(advantages, -10.0, 10.0)

        # PPO update steps
        for _ in range(PPO_EPOCHS):
            with torch.amp.autocast(enabled=True, dtype=torch.float16, device_type="cuda"):
                log_probs, values, dist_entropy = model.evaluate(state_batch, action_batch)
                action_mean, action_std, _ = model(state_batch)
                
                ratio = (log_probs - old_log_probs).exp()
                ratio = torch.clamp(ratio, 0.0, 10.0)
                
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_PARAM, 1 + CLIP_PARAM) * advantages

                # Calculate losses
                actor_loss = -(weights * torch.min(surr1, surr2)).mean()
                critic_loss = 0.5 * (weights * (values - target_values).pow(2)).mean()
                entropy_loss = -0.01 * (weights * dist_entropy).mean()
                
                loss = (actor_loss + critic_loss + entropy_loss) / accumulation_steps

                # Backward pass with gradient accumulation
                scaler.scale(loss).backward()
                
                if (_ + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

        # Update priorities in memory
        priorities = advantages.abs().detach().cpu().numpy()
        memory.update_priorities(indices, priorities)

    # Clear memory after all updates
    memory.clear()

def calculate_bounding_box(robot_x, robot_y, robot_yaw):

    # 机器人中心到边界的相对距离
    half_length = 0.25
    half_width = 0.25

    # 矩形的局部坐标系下的 4 个角点
    corners = np.array([
        [half_length, half_width],
        [half_length, -half_width],
        [-half_length, -half_width],
        [-half_length, half_width]
    ])

    # 旋转矩阵
    rotation_matrix = np.array([
        [np.cos(robot_yaw), -np.sin(robot_yaw)],
        [np.sin(robot_yaw), np.cos(robot_yaw)]
    ])

    # 全局坐标系下的角点
    global_corners = np.dot(corners, rotation_matrix.T) + np.array([robot_x, robot_y])
    return global_corners

def is_point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside

def detect_collision(robot_x, robot_y, robot_yaw, obstacles):
    # 计算边界框
    bounding_box = calculate_bounding_box(robot_x, robot_y, robot_yaw,)

    # 遍历障碍物
    for obstacle in obstacles:
        # 检查是否在边界内
        if is_point_in_polygon(obstacle, bounding_box):
            return True
    return False


def _adjust_dimensions(state_batch, next_state_batch):
    # Remove extra dimension from state_batch if present
    if state_batch.dim() == 5:
        state_batch = state_batch.squeeze(1)
    
    # Remove extra dimension from next_state_batch if present
    if next_state_batch.dim() == 5:
        next_state_batch = next_state_batch.squeeze(1)
        
    # Verify final shapes
    if state_batch.dim() != 4:
        raise ValueError(f"State batch should be 4D, got shape: {state_batch.shape}")
    if next_state_batch.dim() != 4:
        raise ValueError(f"Next state batch should be 4D, got shape: {next_state_batch.shape}")
        
    return state_batch, next_state_batch

def _check_for_nan(tensors, error_message):
    for tensor in tensors:
        if tensor is not None and torch.isnan(tensor).any():
            raise ValueError(error_message)
        
def _check_for_invalid_values(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError(f"[PPO Update] {name} contains invalid values (NaN or Inf).")

def select_action_with_exploration(env, state, model, epsilon, dwa=None, obstacles=None):
    if random.random() < epsilon:
        if dwa is None or obstacles is None:
            raise ValueError("DWA controller or obstacles is not provided")
        print("[Exploration] Using DWA for action generation.")

        robot_x, robot_y, robot_yaw = env.get_robot_position()
        current_speed = env.last_twist.linear.x
        current_omega = env.last_twist.angular.z

        print('robot x = ', robot_x, 'robot_y = ', robot_y)
        state = [robot_x, robot_y, robot_yaw, current_speed, current_omega]
        action, _ = dwa.plan(state, obstacles)  
        action = torch.tensor(action, dtype=torch.float32).to(device)
    else:
        print('action by RL')
        action = model.act(state)
    return action

def save_movement_log_to_csv(movement_log, filename= f"/home/chihsun/catkin_ws/src/my_robot_control/new_waypoint/move_log{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"):
    with open(filename,mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'yaw'])
        for log in movement_log:
            writer.writerow(log)
    print(f"Movement log saved to {filename}")

def main():
    env = GazeboEnv(None)
    dwa = DWA(goal=env.waypoints[env.current_waypoint_index + 3])
    model = ActorCritic(env.observation_space, env.action_space).to(device)
    env.model = model

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.95)  # 每1000次調整學習率
    scaler = GradScaler('cuda')
    memory = PrioritizedMemory(MEMORY_SIZE)

    model_path = "/home/chihsun/catkin_ws/src/my_robot_control/scripts/saved_model_ppo.pth"
    best_model_path = "/home/chihsun/catkin_ws/src/my_robot_control/scripts/best_model.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Loaded existing model.")
    else:
        print("Created new model.")
    
    env.visualize_original_path()

    num_episodes = 1000000
    best_test_reward = -np.inf
    
    last_recorded_position = None

    # Initialize obstacles
    static_obstacles = []
    for y in range(env.slam_map.shape[0]):
        for x in range(env.slam_map.shape[1]):
            if env.slam_map[y, x] < 190:
                ox, oy = env.image_to_gazebo_coords(x, y)
                static_obstacles.append((ox, oy))
 
    for e in range(num_episodes):
        movement_log = []
        if not env.optimized_waypoints_calculated:
            env.optimize_waypoints_with_a_star()

        state = env.reset()
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.clone().detach().unsqueeze(0).to(device)

        total_reward = 0
        start_time = time.time()
        use_rl = False  # Flag to determine if RL is used in this episode

        for time_step in range(1500):
            robot_x, robot_y, robot_yaw = env.get_robot_position()

            if last_recorded_position is None or np.linalg.norm(
                [robot_x - last_recorded_position[0], robot_y - last_recorded_position[1]]
            ) >= 1.0386:
                movement_log.append((robot_x, robot_y, robot_yaw))
                last_recorded_position = (robot_x, robot_y)

            obstacles = [
                (ox, oy) for ox, oy in static_obstacles
                if np.sqrt((ox-robot_x)**2 + (oy - robot_y)**2) < 4.0
            ]
            obstacles = grid_filter(obstacles, grid_size=0.7)

            lookahead_index = min(env.current_waypoint_index + 3, len(env.waypoint_distances)-1)
            dwa.goal = env.waypoints[lookahead_index]

            failure_range = range(
                max(0, env.current_waypoint_index - 6),
                min(len(env.waypoints), env.current_waypoint_index + 2)
            )
            failure_counts = {i: env.waypoint_failures.get(i, 0) for i in failure_range}

            use_deep_rl_control = any(
                env.waypoint_failures.get(i, 0) > 1 for i in failure_range
            )

            if use_deep_rl_control:
                use_rl = True  # Mark this episode as using RL
                action = select_action_with_exploration(env, state, model, epsilon=0.6 , dwa=dwa, obstacles=obstacles)
                action_np = action.detach().cpu().numpy().flatten()
                print(f"RL Action at waypoint {env.current_waypoint_index}: {action_np}")
            else:
                action_np = env.calculate_action_pure_pursuit()
                print(f"A* Action at waypoint {env.current_waypoint_index}: {action_np}")

            next_state, reward, done, _ = env.step(action_np, obstacles=obstacles)

            if not isinstance(next_state, torch.Tensor):
                next_state = torch.tensor(next_state, dtype=torch.float32)
            next_state = next_state.clone().detach().unsqueeze(0).to(device)

            if use_deep_rl_control:
                memory.add(state.cpu().numpy(), action_np, reward, done, next_state.cpu().numpy())
                total_reward += reward

            state = next_state

            state = (state - state.min()) / (state.max() - state.min() + 1e-5)

            elapsed_time = time.time() - start_time
            if done or elapsed_time > 240:
                if elapsed_time > 240:
                    reward -= 10.0
                    print(f"Episode {e} failed at time step {time_step}: time exceeded 240 sec.")
                break

        if use_rl and len(memory.memory) > BATCH_SIZE:
            print(123456)
            current_batch_size = min(BATCH_SIZE, len(memory.memory))
            ppo_update(PPO_EPOCHS, env, model, optimizer, memory, scaler, batch_size=current_batch_size,scheduler=scheduler)

        print(f"Episode {e}, Total Reward: {total_reward}, LR: {scheduler.get_last_lr()[0]}")

        if total_reward > best_test_reward:
            best_test_reward = total_reward
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with reward: {best_test_reward}")
            # save_movement_log_to_csv(movement_log)

        if e % 5 == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved after {e} episodes.")
        rospy.sleep(1.0)

    torch.save(model.state_dict(), model_path)
    print("Final model saved.")


if __name__ == '__main__':
    main()
