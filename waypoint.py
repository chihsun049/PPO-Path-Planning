#!/usr/bin/env python3
import rospy
import numpy as np
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from scipy.spatial import KDTree
from PIL import Image
import yaml
import time
from skimage.draw import line

class PathOptimizer:
    def __init__(self):
        self.target_x = -7.2213
        self.target_y = -1.7003
        self.waypoints = self.generate_waypoints()
        self.optimized_waypoints = []
        self.waypoint_distances = self.calculate_waypoint_distances()   # 計算一整圈機器人要走的大致距離
        self.total_path_distance = sum(self.waypoint_distances)  # 計算總路徑距離
        self.current_waypoint_index = 0

        # 加载 SLAM 地图
        self.load_slam_map('/home/chihsun/catkin_ws/src/my_robot_control/scripts/my_map0924.yaml')

        # 初始化结果保存路径
        self.result_folder = '/home/chihsun/catkin_ws/src/my_robot_control/scripts/optimized_path_img/'
        os.makedirs(self.result_folder, exist_ok=True)
        self.result_metrics_file = os.path.join(self.result_folder, 'path_metrics.csv')
    
    def reset_state(self):
        """
        重置路徑規劃的狀態變數，每次規劃新路徑前需要調用。
        """
        self.target_x = -7.2213
        self.target_y = -1.7003
        self.waypoints = self.generate_waypoints()
        self.optimized_waypoints = []
        self.waypoint_distances = self.calculate_waypoint_distances()  # 計算路徑點之間的距離
        self.total_path_distance = sum(self.waypoint_distances)       # 總路徑距離
        self.current_waypoint_index = 0                               # 當前路徑點索引
        self.g_scores = {}                                            # g 值累積記錄
        self.obstacle_distances = []  # 儲存所有路徑點到障礙物的距離

    def load_slam_map(self, yaml_path):
        with open(yaml_path, 'r') as file:
            map_metadata = yaml.safe_load(file)
            png_path = map_metadata['image'].replace(".pgm", ".png")
            png_image = Image.open(png_path).convert('L')
            self.slam_map = np.array(png_image)

        self.generate_costmap()

    def generate_costmap(self):
        if self.slam_map is None:
            rospy.logerr("SLAM map not loaded. Cannot generate costmap.")
            return False

        wall_color = np.array([100, 100, 100])
        wall_color2 = np.array([120, 120, 120])
        
        img = cv2.cvtColor(self.slam_map, cv2.COLOR_GRAY2BGR)
        wall_mask = cv2.inRange(img, wall_color, wall_color2)
        
        self.cost_map = np.zeros_like(self.slam_map)
        
        inner_dilation = 3
        outer_dilation = 6
        
        inner_kernel = np.ones((inner_dilation * 2 + 1, inner_dilation * 2 + 1), np.uint8)
        inner_dilated = cv2.dilate(wall_mask, inner_kernel, iterations=1)
        
        outer_kernel = np.ones((outer_dilation * 2 + 1, outer_dilation * 2 + 1), np.uint8)
        outer_dilated = cv2.dilate(wall_mask, outer_kernel, iterations=1)
        
        outer_only = cv2.subtract(outer_dilated, inner_dilated)
        
        self.cost_map[wall_mask > 0] = 254        # 障礙物設為最高代價
        self.cost_map[inner_dilated > 0] = 190    # 內層膨脹區設為中高代價
        self.cost_map[outer_only > 0] = 100        # 外層膨脹區設為中低代價
        
        rospy.loginfo("Costmap generated successfully.")
        return True

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
            (self.target_x, self.target_y)]
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

    def is_line_free(self, png_image, current, neighbor, safe_threshold=230):
        current = np.array(current, dtype=np.int32)
        neighbor = np.array(neighbor, dtype=np.int32)

        # 使用 Bresenham 算法生成線段
        rr, cc = line(current[1], current[0], neighbor[1], neighbor[0])

        # 檢查是否越界
        if np.any((rr < 0) | (rr >= png_image.shape[0]) | (cc < 0) | (cc >= png_image.shape[1])):
            return False

        # 檢查線段上的像素是否有障礙物
        if np.any(png_image[rr, cc] < safe_threshold):
            return False

        return True

    def calculate_min_distance_to_obstacles(self, x, y, kd_tree):
        """
        計算給定點 (x, y) 到最近障礙物的距離。
        """
        if kd_tree.data.size == 0:  # 確保 KDTree 中有障礙物數據
            rospy.logwarn("KDTree is empty. Cannot calculate obstacle distances.")
            return float('inf')  # 返回無窮大表示無障礙物

        distance, _ = kd_tree.query((x, y))
        return distance

    def a_star_optimize_waypoint(
        self, png_image, start_point, goal_point, kd_tree, 
        g_weight=1, h_weight=1, smoothness_weight=1, distance_penalty_weight=1, grid_size=50):
        if not hasattr(self, 'g_scores'):
            self.g_scores = {}  # 初始化 g_scores 属性，用于保存跨路径点的累积距离

        img_start_x, img_start_y = self.gazebo_to_image_coords(*start_point)
        img_goal_x, img_goal_y = self.gazebo_to_image_coords(*goal_point)

        # 初始化起点 g 值，如果不存在则设为 0
        if (img_start_x, img_start_y) not in self.g_scores:
            self.g_scores[(img_start_x, img_start_y)] = 0

        # 获取当前参考点的索引
        current_wp_index = self.current_waypoint_index

        # 确保索引范围有效
        if current_wp_index >= len(self.waypoint_distances):
            rospy.logerr("Waypoint index out of range for h normalization.")
            return start_point  # 如果发生错误，返回起点

        # 分母从 self.waypoint_distances 获取当前参考点到目标参考点的距离
        h_normalization_denominator = self.waypoint_distances[current_wp_index]

        # 初始化
        best_f_score = float('inf')
        best_point = (img_start_x, img_start_y)

        # 獲取範圍內所有候選點
        candidate_points = [
            (xi, yi)
            for xi in range(img_start_x - grid_size // 2, img_start_x + grid_size // 2)
            for yi in range(img_start_y - grid_size // 2, img_start_y + grid_size // 2)
            if 0 <= xi < png_image.shape[1] and 0 <= yi < png_image.shape[0]
        ]

        # 查詢所有候選點到障礙物的距離
        if candidate_points:
            distances = kd_tree.query(candidate_points)[0]  # 距離列表
            max_obstacle_distance = np.max(distances)  # 最大距離
        else:
            max_obstacle_distance = 1  # 避免分母為 0

        for x, y in candidate_points:
            # 检查从上一个路径点到当前候选点是否通畅
            if self.optimized_waypoints:
                prev_point = self.optimized_waypoints[-1]
                prev_img_x, prev_img_y = self.gazebo_to_image_coords(*prev_point)
                if not self.is_line_free(png_image, (prev_img_x, prev_img_y), (x, y)):
                    continue  # 如果有障碍物，跳过该点

            # 计算代价地图权重
            costmap_cost = self.cost_map[y, x]

            # 当前点的移动距离计算基于车辆的实际路径点
            if len(self.optimized_waypoints) > 0:
                last_waypoint = self.optimized_waypoints[-1]
                last_img_x, last_img_y = self.gazebo_to_image_coords(*last_waypoint)
                step_distance = np.sqrt((x - last_img_x) ** 2 + (y - last_img_y) ** 2)
                g = self.g_scores.get((last_img_x, last_img_y), 0) + step_distance
            else:
                step_distance = 0
                g = self.g_scores[(img_start_x, img_start_y)]

            self.g_scores[(x, y)] = g

            # 正規化 g 值
            g_normalized = (g * 0.05) / self.total_path_distance

            # 当前点到目标点的 h 值（启发式）
            h = np.sqrt((x - img_goal_x) ** 2 + (y - img_goal_y) ** 2)

            # 正規化 h 值，使用當前點到參考點的距離作為分母
            h_normalized = (h * 0.05) / h_normalization_denominator if h_normalization_denominator > 0 else h

            # 平滑性代价调整
            if len(self.optimized_waypoints) >= 2:
                prev_prev_point = self.optimized_waypoints[-2]
                prev_prev_img_x, prev_prev_img_y = self.gazebo_to_image_coords(*prev_prev_point)
                prev_point = self.optimized_waypoints[-1]
                prev_img_x, prev_img_y = self.gazebo_to_image_coords(*prev_point)

                # 差分计算
                delta_xi = (prev_img_x - prev_prev_img_x, prev_img_y - prev_prev_img_y)
                delta_xi1 = (x - prev_img_x, y - prev_img_y)
                smoothness_cost = (delta_xi1[0] - delta_xi[0]) ** 2 + (delta_xi1[1] - delta_xi[1]) ** 2

                # 正規化平滑性代價
                max_smoothness_cost = grid_size ** 2  # 假设最大位移为 grid_size 的平方和
                smoothness_cost_normalized = smoothness_cost / max_smoothness_cost if max_smoothness_cost > 0 else 0
            else:
                smoothness_cost_normalized = 0

            obstacle_distance = self.calculate_min_distance_to_obstacles(x, y, kd_tree)

            # 如果障礙物距離為有效值，加入累積列表
            if obstacle_distance != float('inf'):
                self.obstacle_distances.append(obstacle_distance)

            # 正規化距离代价
            distance_penalty_normalized = -obstacle_distance / max_obstacle_distance if max_obstacle_distance > 0 else 0

            # 计算总的代价 f
            f = (
                (g_normalized * g_weight + h_normalized * h_weight) * 0.34 +
                smoothness_cost_normalized * smoothness_weight * 0.33 + distance_penalty_normalized * distance_penalty_weight * 0.33 +
                costmap_cost
            )

            if f < best_f_score:
                best_f_score = f
                best_point = (x, y)

        optimized_gazebo_x, optimized_gazebo_y = self.image_to_gazebo_coords(*best_point)
        return optimized_gazebo_x, optimized_gazebo_y

    def optimize_waypoints_with_a_star(self, g_weight, smoothness_weight, distance_penalty_weight):
        # 每次規劃新路徑前重置狀態
        self.reset_state()
        
        obstacle_points = [
            (x, y) for y in range(self.slam_map.shape[0]) for x in range(self.slam_map.shape[1])
            if self.slam_map[y, x] < 250
        ]
        kd_tree = KDTree(obstacle_points)

        for i in range(len(self.waypoints) - 1):
            self.current_waypoint_index = i  # 更新當前路徑點索引

            start_point = self.waypoints[i]
            goal_point = self.waypoints[i + 1]

            optimized_point = self.a_star_optimize_waypoint(
                self.slam_map, start_point, goal_point, kd_tree,
                g_weight=g_weight,
                h_weight=1,
                smoothness_weight=smoothness_weight,
                distance_penalty_weight=distance_penalty_weight
            )
            self.optimized_waypoints.append(optimized_point)

        self.optimized_waypoints.append(self.waypoints[-1])  # 添加終點
    
    def visualize_complete_path(self, waypoints, save_path):
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

        # 添加圖例和標題
        plt.legend(fontsize=12)
        plt.title('Path Visualization with Cost Map', fontsize=14)
        
        # 設置軸的範圍
        plt.xlim(0, cost_map_rgb.shape[1])
        plt.ylim(cost_map_rgb.shape[0], 0)  # 注意：圖像坐標 y 軸是倒置的

        # 保存圖片
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        rospy.loginfo(f"Path visualization with cost map saved to {save_path}")

    def evaluate_path_metrics(self):
        total_length = 0.0
        smoothness = 0.0

        for i in range(len(self.optimized_waypoints) - 1):
            x1, y1 = self.optimized_waypoints[i]
            x2, y2 = self.optimized_waypoints[i + 1]
            total_length += np.linalg.norm([x2 - x1, y2 - y1])

            if i < len(self.optimized_waypoints) - 2:
                x3, y3 = self.optimized_waypoints[i + 2]
                v1 = np.array([x2 - x1, y2 - y1])
                v2 = np.array([x3 - x2, y3 - y2])

                # 確保向量長度非零，避免數學錯誤
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                if norm_v1 > 0 and norm_v2 > 0:
                    angle = np.arccos(
                        np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
                    )
                    smoothness += angle ** 2
                else:
                    rospy.logwarn(f"Zero-length vector encountered at index {i}. Skipping smoothness calculation for this segment.")

        # 計算平均障礙物距離
        avg_obstacle_distance = (
            np.mean(self.obstacle_distances) if self.obstacle_distances else float('inf')
        )

        return total_length, avg_obstacle_distance, smoothness

def main():
    optimizer = PathOptimizer()

    # 定义权重范围
    g_weights = range(1, 11)  # g_weight 从 1 到 10
    smoothness_weights = range(1, 11)  # smoothness_weight 从 1 到 10
    distance_penalty_weights = range(1, 11)  # distance_penalty_weight 从 1 到 10

    results = []
    for g_weight in g_weights:
        for smoothness_weight in smoothness_weights:
            for distance_penalty_weight in distance_penalty_weights:
                # 优化路径
                optimizer.optimize_waypoints_with_a_star(
                    g_weight=g_weight,
                    smoothness_weight=smoothness_weight,
                    distance_penalty_weight=distance_penalty_weight
                )
                
                # 计算路径指标
                total_length, avg_obstacle_distance, smoothness = optimizer.evaluate_path_metrics()
                results.append({
                    "g_weight": g_weight,
                    "smoothness_weight": smoothness_weight,
                    "distance_penalty_weight": distance_penalty_weight,
                    "total_length": total_length,
                    "avg_obstacle_distance": avg_obstacle_distance,
                    "smoothness": smoothness
                })

                # 保存每条优化路径的可视化图片
                save_path = os.path.join(
                    optimizer.result_folder,
                    f"optimized_path_g{g_weight}_s{smoothness_weight}_d{distance_penalty_weight}.png"
                )
                optimizer.visualize_complete_path(optimizer.optimized_waypoints, save_path=save_path)

    # 保存分析结果
    df = pd.DataFrame(results)
    df.to_csv(optimizer.result_metrics_file, index=False)

    # 可视化分析结果（根据需求选择展示的指标）
    for metric in ["total_length", "avg_obstacle_distance", "smoothness"]:
        plt.figure(figsize=(12, 6))
        for g_weight in g_weights:
            subset = df[df["g_weight"] == g_weight]
            plt.plot(
                subset["distance_penalty_weight"] + subset["smoothness_weight"] * 10, 
                subset[metric], label=f"g_weight={g_weight}"
            )
        plt.xlabel("Combined Weight Index (distance_penalty + smoothness * 10)")
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.title(f"Path {metric.replace('_', ' ').title()} vs. Weight Combinations")
        plt.savefig(os.path.join(optimizer.result_folder, f"{metric}_vs_weights.png"))
        plt.close()

if __name__ == "__main__":
    main()
