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
import yaml
from PIL import Image
from skimage.draw import line
import matplotlib.pyplot as plt

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
        states = [s if s.dim() == 4 else s.view(1, 3, 64, 64) for s in states]
        next_states = [ns if ns.dim() == 4 else ns.view(1, 3, 64, 64) for ns in next_states]

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

        self.max_no_progress_steps = 10
        self.no_progress_steps = 0
        self.use_rl_for_a_star = False  # 是否啟用 RL 介入 A*
        self.episode_counter = 0  # 用於控制模仿學習階段與 RL 過渡

        self.subpath_stability = {}  # 追蹤子路徑穩定性
        self.waypoint_failures = {i: 0 for i in range(len(self.waypoints))}  # 初始化失敗次數
        
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
    
    def heuristic_cost(self, current, goal, next_waypoint, waypoint_list, waypoint_index, direction_weight=5.0, obstacle_weight=1.0, dist_to_goal_weight=1.0):
        """
        综合启发函数，明确引导路径规划顺序。
        - current: 当前点 (像素坐标)。
        - goal: 当前子目标点 (像素坐标)。
        - next_waypoint: 下一个目标路径点 (像素坐标)。
        - waypoint_list: 路径点列表。
        - waypoint_index: 当前路径点索引。
        """
        # 距离目标点的直线距离
        dist_to_goal = np.linalg.norm(np.array(goal) - np.array(current))

        # 距离障碍物的惩罚
        obstacle_distance = self.calculate_obstacle_distance(current)
        obstacle_penalty = max(0, 10 / (obstacle_distance + 1e-6) - 1) if obstacle_distance < 10 else 0

        # 引导方向权重：偏离路径序列的方向会增加惩罚
        current_to_next = np.array(next_waypoint) - np.array(current)
        waypoint_direction = np.array(waypoint_list[waypoint_index + 1]) - np.array(next_waypoint)

        # 计算当前点到下一个路径点的方向误差
        angle_error = np.arccos(
            np.clip(
                np.dot(current_to_next, waypoint_direction) / 
                (np.linalg.norm(current_to_next) * np.linalg.norm(waypoint_direction) + 1e-6),
                -1.0, 1.0
            )
        )
        direction_penalty = angle_error

        # 综合启发值：距离、方向、障碍物
        return (
            dist_to_goal_weight * dist_to_goal +
            direction_weight * direction_penalty +
            obstacle_weight * obstacle_penalty
        )
    
    def calculate_obstacle_distance(self, point):
        # 找到距離點最近的障礙物（以像素計算）
        x, y = point
        search_range = 10  # 搜索範圍（像素）
        obstacle_coords = np.argwhere(self.slam_map[max(0, y - search_range):min(self.slam_map.shape[0], y + search_range),
                                                    max(0, x - search_range):min(self.slam_map.shape[1], x + search_range)] < 250)
        if len(obstacle_coords) == 0:
            return float('inf')  # 如果周圍無障礙物，返回無窮遠
        obstacle_coords += [max(0, x - search_range), max(0, y - search_range)]  # 還原全局座標
        distances = np.linalg.norm(obstacle_coords - np.array([x, y]), axis=1)
        return distances.min()  # 返回最近障礙物的距離
    
    def get_neighbors(self, current, step=1):
        """
        生成邻居点。
        - current: 当前点 (x, y)。
        - step: 固定步长。
        """
        x, y = current
        directions = [
            (step, 0), (-step, 0), (0, step), (0, -step),  # 上下左右
            (step, step), (step, -step), (-step, step), (-step, -step)  # 对角线
        ]
        neighbors = [
            (x + dx, y + dy)
            for dx, dy in directions
            if 0 <= x + dx < self.slam_map.shape[1] and 0 <= y + dy < self.slam_map.shape[0]
        ]
        return neighbors
    
    def reconstruct_path(self, came_from, current):
        """
        回溯來重建完整的路徑。
        """
        path = [current]
        while current in came_from:
            next_node = came_from[current]
            if np.linalg.norm(np.array(current) - np.array(next_node)) < 1e-3:  # 容錯處理
                break
            current = next_node
            path.append(current)
        path.reverse()
        return path
    
    def is_line_free(self, png_image, start, end, safe_threshold=250):
        """
        檢查從 start 到 end 的直線是否沒有障礙物。

        :param png_image: SLAM 地圖的二值化圖片。
        :param start: 起點座標 (x, y)。
        :param end: 終點座標 (x, y)。
        :param safe_threshold: 判定障礙物的閾值（像素值低於此值視為障礙物）。
        :return: True 如果直線上沒有障礙物，否則返回 False。
        """
        # 確保座標為整數
        start = (int(round(start[0])), int(round(start[1])))
        end = (int(round(end[0])), int(round(end[1])))

        # 使用 Bresenham 算法生成線上的像素點
        rr, cc = line(start[1], start[0], end[1], end[0])  # skimage.draw.line

        # 遍歷線上的每個點檢查障礙物
        for r, c in zip(rr, cc):
            # 檢查是否越界
            if not (0 <= r < png_image.shape[0] and 0 <= c < png_image.shape[1]):
                return False  # 超出地圖範圍視為障礙
            # 檢查像素值是否低於閾值
            if png_image[r, c] < safe_threshold:
                return False  # 發現障礙物
        return True  # 全部點均可通行

    def dynamic_grid_size(self, current, neighbor):
        obstacle_distance = self.calculate_obstacle_distance(current)
        if obstacle_distance > 50:  # 遠離障礙物
            return 30
        elif obstacle_distance > 30:  # 中等距離
            return 20
        elif obstacle_distance > 10:  # 中等距離
            return 5
        else:  # 靠近障礙物
            return 1
    
    def visualize_path(self, start, goal, path_points, map_path='/home/chihsun/catkin_ws/src/my_robot_control/scripts/my_map0924.png', save_path='/home/chihsun/catkin_ws/src/my_robot_control/scripts/a_star_path.png'):
        """
        可视化 A* 生成的路径，并将其保存为图片。
        :param start: 起点坐标 (gazebo 坐标系)
        :param goal: 终点坐标 (gazebo 坐标系)
        :param path_points: A* 生成的路径点列表 (像素坐标系)
        :param map_path: SLAM 地图路径
        :param save_path: 保存图片的路径
        """
        # 加载 SLAM 地图
        if not hasattr(self, 'slam_map'):
            raise ValueError("SLAM map not loaded.")
        
        # 转换地图为灰度图
        map_img = self.slam_map.copy()
        map_img[map_img < 250] = 0  # 障碍物区域
        map_img[map_img >= 250] = 255  # 可通行区域

        # 绘制地图
        plt.figure(figsize=(10, 10))
        plt.imshow(map_img, cmap='gray', origin='upper')

        # 转换路径点到图像坐标
        img_start = self.gazebo_to_image_coords(*start)
        img_goal = self.gazebo_to_image_coords(*goal)
        img_path = [(p[0], p[1]) for p in path_points]

        # 绘制起点、终点和路径
        plt.scatter(img_start[0], img_start[1], color='red', label='Start', s=50)
        plt.scatter(img_goal[0], img_goal[1], color='blue', label='Goal', s=50)
        path_x, path_y = zip(*img_path)
        plt.plot(path_x, path_y, color='green', linewidth=2, label='A* Path')

        # 添加图例和保存图片
        plt.legend()
        plt.title('A* Path Visualization')
        plt.savefig(save_path)
        plt.close()
        rospy.loginfo(f"Path visualization saved to {save_path}")
    
    def visualize_complete_path(self, complete_path, save_path='/home/chihsun/catkin_ws/src/my_robot_control/scripts/full_path.png'):
        """
        可视化完整路径，并将其保存为图片。
        :param complete_path: 完整路径点列表（像素坐标系）。
        :param save_path: 保存图片的路径。
        """
        if not hasattr(self, 'slam_map'):
            raise ValueError("SLAM map not loaded.")
        
        # 转换地图为灰度图
        map_img = self.slam_map.copy()
        map_img[map_img < 250] = 0  # 障碍物区域
        map_img[map_img >= 250] = 255  # 可通行区域

        # 绘制地图
        plt.figure(figsize=(10, 10))
        plt.imshow(map_img, cmap='gray', origin='upper')

        # 转换路径点到图像坐标
        img_path = [(p[0], p[1]) for p in complete_path]
        path_x, path_y = zip(*img_path)

        # 绘制路径
        plt.plot(path_x, path_y, color='green', linewidth=2, label='Full Path')

        # 标注起点和终点
        img_start = self.gazebo_to_image_coords(*self.waypoints[0])
        img_goal = self.gazebo_to_image_coords(*self.waypoints[-1])
        plt.scatter(img_start[0], img_start[1], color='red', label='Start', s=50)
        plt.scatter(img_goal[0], img_goal[1], color='blue', label='Goal', s=50)

        # 添加图例和保存图片
        plt.legend()
        plt.title('Complete Path Visualization')
        plt.savefig(save_path)
        plt.close()
        rospy.loginfo(f"Complete path visualization saved to {save_path}")

    def a_star_optimize_waypoint(self, png_image, start_point, goal_point, waypoint_list, waypoint_index, step=1):
        img_start_x, img_start_y = self.gazebo_to_image_coords(*start_point)
        img_goal_x, img_goal_y = self.gazebo_to_image_coords(*goal_point)

        open_set = [(img_start_x, img_start_y)]
        came_from = {}
        g_score = {open_set[0]: 0}
        f_score = {open_set[0]: self.heuristic_cost(open_set[0], (img_goal_x, img_goal_y), waypoint_list[waypoint_index + 1], waypoint_list, waypoint_index)}

        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))

            if current == (img_goal_x, img_goal_y):  # 到达当前目标点
                path = self.reconstruct_path(came_from, current)
                self.visualize_path(start_point, goal_point, path)  # 可视化路径
                return path

            open_set.remove(current)
            for neighbor in self.get_neighbors(current, step):
                if not self.is_line_free(png_image, current, neighbor):  # 检查直线通行性
                    continue

                tentative_g_score = g_score[current] + np.linalg.norm(np.array(current) - np.array(neighbor))
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = (
                        tentative_g_score +
                        self.heuristic_cost(neighbor, (img_goal_x, img_goal_y), waypoint_list[waypoint_index + 1], waypoint_list, waypoint_index)
                    )
                    if neighbor not in open_set:
                        open_set.append(neighbor)

        rospy.logwarn(f"A* failed between {start_point} and {goal_point}.")
        return [start_point, goal_point]

    def optimize_waypoints_with_a_star(self):
        if self.optimized_waypoints_calculated:
            rospy.loginfo("Optimized waypoints already calculated. Skipping.")
            return

        rospy.loginfo("Starting waypoint optimization with A*...")
        optimized_waypoints = [self.waypoints[0]]  # 初始化路径，包含起点
        complete_path = []  # 用于存储完整的路径点

        for i in range(len(self.waypoints) - 1):
            start = self.waypoints[i]
            goal = self.waypoints[i + 1]

            rospy.loginfo(f"Optimizing path segment {i}: Start {start} -> Goal {goal}")

            # 调用 A* 优化路径段
            path_segment = self.a_star_optimize_waypoint(self.slam_map, start, goal, self.waypoints, i)

            # 如果路径生成失败，保留直接连接路径点
            if len(path_segment) <= 2:
                rospy.logwarn(f"A* failed for segment {i}. Using direct connection.")
                optimized_waypoints.append(goal)
            else:
                optimized_waypoints.extend(path_segment[1:])  # 跳过重复的起点
                complete_path.extend(path_segment)  # 将路径段添加到完整路径中

        self.optimized_waypoints = optimized_waypoints
        self.waypoints = optimized_waypoints  # 更新为优化后的路径点
        self.optimized_waypoints_calculated = True

        # 在优化完成后保存完整路径图像
        self.visualize_complete_path(complete_path)
        rospy.loginfo("Waypoint optimization complete.")
    
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

        # 打印調試訊息
        print(f"Robot position: ({robot_x}, {robot_y})")
        print(f"Closest waypoint index: {closest_index}, Distance to closest waypoint: {distances[closest_index]}")
        print(f"Current waypoint index: {self.current_waypoint_index}")

        # 如果找到更近的路徑點，更新 current_waypoint_index 並給予獎勵
        if closest_index > self.current_waypoint_index:
            distance_reward = sum(self.waypoint_distances[self.current_waypoint_index:closest_index])
            reward += distance_reward * 100
            self.current_waypoint_index = closest_index
            print('Distance to goal reward:', reward)

        # 判斷是否達到目標點
        distance_to_goal = distances[closest_index]
        if self.current_waypoint_index == len(self.waypoints) - 1 and distance_to_goal < REFERENCE_DISTANCE_TOLERANCE:
            print("Goal reached!")
            reward += 1000  # 給予額外獎勵
            return self.state, reward, True, {}

        # 計算行動
        action = self.calculate_action_pure_pursuit()
        linear_speed = np.clip(action[0], -2.0, 3.0)
        steer_angle = np.clip(action[1], -0.6, 0.6)

        # 發布速度指令
        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = steer_angle
        self.pub_cmd_vel.publish(twist)

        # 更新 IMU 資料
        imu_data = self.generate_imu_data()
        self.pub_imu.publish(imu_data)

        rospy.sleep(0.1)

        # 計算獎勵（例如考慮障礙物或路徑平滑性）
        reward, done = self.calculate_reward(robot_x, robot_y, reward, self.state)
        return self.state, reward, done, {}

    def reset(self):
        robot_x, robot_y, _ = self.get_robot_position()
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
        if not self.waypoints or any(wp is None for wp in self.waypoints):
            rospy.logwarn("Waypoints are invalid during reset. Regenerating.")
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

        # 確保狀態是4D張量
        if isinstance(self.state, np.ndarray):
            self.state = torch.tensor(self.state, dtype=torch.float32).view(1, 3, 64, 64).to(device)
        elif self.state.dim() != 4:
            self.state = self.state.view(1, 3, 64, 64)

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
        obstacle_count = np.sum(occupancy_grid <= 190)  # 假設state[0]為佔據網格通道
        reward += 300 - obstacle_count*3

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
    def __init__(self, input_dim, action_dim):  # 確保 input_dim 與展平後的狀態匹配
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # input_dim 是展平後的維度
        self.fc2 = nn.Linear(128, 128)

        # Actor 和 Critic
        self.actor = nn.Linear(128, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))  # 標準差參數
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平成 (batch_size, input_dim)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_mean = self.actor(x)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        value = self.critic(x)
        return action_mean, action_std, value

    def act(self, state):
        action_mean, action_std, _ = self(state)
        action = action_mean + action_std * torch.randn_like(action_std)  # 採樣行動
        return action.tanh()  # 使用 tanh 限制動作範圍

    def evaluate(self, state, action):
        action_mean, action_std, value = self(state)
        dist = torch.distributions.Normal(action_mean, action_std)
        action_log_probs = dist.log_prob(action).sum(dim=-1, keepdim=True)  # 動作的對數概率
        dist_entropy = dist.entropy().sum(dim=-1, keepdim=True)  # 熵正則
        return action_log_probs, value, dist_entropy

def ppo_update(ppo_epochs, env, model, optimizer, memory, scaler):
    for _ in range(ppo_epochs):
        state_batch, action_batch, reward_batch, done_batch, next_state_batch, indices, weights = memory.sample(BATCH_SIZE)
        
        adjusted_lr = LEARNING_RATE * (weights.mean().item() + 1e-3)
        for param_group in optimizer.param_groups:
            param_group['lr'] = adjusted_lr

        with torch.no_grad():
            old_log_probs, _, _ = model.evaluate(state_batch.view(state_batch.size(0), -1), action_batch)
        old_log_probs = old_log_probs.detach()

        for _ in range(PPO_EPOCHS):
            with torch.amp.autocast('cuda'):
                log_probs, state_values, dist_entropy = model.evaluate(state_batch, action_batch)

                # 確保 reward_batch 和 state_values 都具有形狀 (batch_size, 1)
                reward_batch = reward_batch.unsqueeze(-1) if reward_batch.dim() == 1 else reward_batch
                done_batch = done_batch.unsqueeze(-1) if done_batch.dim() == 1 else done_batch
                state_values = state_values.unsqueeze(-1) if state_values.dim() == 1 else state_values

                # 計算 advantages
                with torch.no_grad():
                    next_state_values = model(next_state_batch)[2]
                    next_state_values = next_state_values.unsqueeze(-1) if next_state_values.dim() == 1 else next_state_values
                
                target_values = reward_batch + (1 - done_batch) * GAMMA * next_state_values
                advantages = target_values - state_values

                # 計算 PPO 損失
                ratio = (log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_PARAM, 1 + CLIP_PARAM) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(state_values, target_values)
                entropy_loss = -0.01 * dist_entropy.mean()  # 添加熵正則項
                loss = actor_loss + 0.5 * critic_loss + entropy_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            priorities = (advantages + 1e-5).abs().detach().cpu().numpy()
            memory.update_priorities(indices, priorities)

def main():
    # 初始化環境
    env = GazeboEnv(None)

    input_dim = np.prod(env.observation_space)  # 確保與展平後的維度一致
    action_dim = env.action_space
    model = ActorCritic(input_dim=input_dim, action_dim=action_dim).to(device)
    env.model = model

    # 初始化優化器和梯度縮放器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler('cuda')
    memory = PrioritizedMemory(MEMORY_SIZE)

    # 模型保存路徑
    model_path = "/home/chihsun/catkin_ws/src/my_robot_control/scripts/saved_model_ppo.pth"
    best_model_path = "/home/chihsun/catkin_ws/src/my_robot_control/scripts/best_model.pth"
    
    # 加載已有的模型（如果存在）
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Loaded existing model.")
    else:
        print("Created new model.")

    # 訓練參數
    num_episodes = 1000000
    imitation_learning_episodes = 100  # 模仿學習的回合數
    best_test_reward = -np.inf

    # 訓練循環
    for e in range(num_episodes):
        # 更新階段進度
        env.episode_counter = e

        if e < imitation_learning_episodes:
            print(f"Episode {e}: Imitation learning phase.")
            env.use_rl_for_a_star = False  # 完全使用 A*
        else:
            print(f"Episode {e}: Mixed RL phase.")
            env.use_rl_for_a_star = True  # 混合 RL

        # 確保優化過的路徑點被使用
        if not env.optimized_waypoints_calculated:
            env.optimize_waypoints_with_a_star()

        # 重置環境
        state = env.reset()
        if not isinstance(state, torch.Tensor):  # 確保狀態是張量
            state = torch.tensor(state, dtype=torch.float32)
        state = state.clone().detach().unsqueeze(0).to(device)

        total_reward = 0
        start_time = time.time()

        # 時間步進
        for time_step in range(1500):
            # 使用模型計算動作
            state_flattened = state.view(state.size(0), -1)  # 將狀態展平
            action = model.act(state_flattened)
            action_np = action.detach().cpu().numpy()

            # 執行環境一步
            next_state, reward, done, _ = env.step(action_np)

            # 確保 next_state 是張量
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.tensor(next_state, dtype=torch.float32)
            next_state = next_state.clone().detach().unsqueeze(0).to(device)

            # 將數據存入記憶體
            memory.add(state.cpu().numpy(), action_np, reward, done, next_state.cpu().numpy())

            # 更新 state 和總分
            state = next_state
            total_reward += reward

            # 檢查是否達到終止條件
            elapsed_time = time.time() - start_time
            if done or elapsed_time > 240:
                if elapsed_time > 240:
                    reward -= 1000.0
                    print(f"Episode {e} failed at time step {time_step}: time exceeded 240 sec.")
                break

        # 更新 PPO 模型
        ppo_update(PPO_EPOCHS, env, model, optimizer, memory, scaler)
        memory.clear()

        # 打印回合信息
        print(f"Episode {e}, Total Reward: {total_reward}")

        # 保存最佳模型
        if total_reward > best_test_reward:
            best_test_reward = total_reward
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with reward: {best_test_reward}")

        # 每五次保存一次模型
        if e % 5 == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved after {e} episodes.")

        rospy.sleep(1.0)

    # 保存最終模型
    torch.save(model.state_dict(), model_path)
    print("Final model saved.")

if __name__ == '__main__':
    main()