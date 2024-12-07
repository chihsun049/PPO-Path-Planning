#!/usr/bin/env python3
import rospy
import numpy as np
from scipy.spatial import KDTree
import yaml
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time
from datetime import datetime

class PathOptimizer:
    def __init__(self):
        # 初始化ROS節點
        rospy.init_node('path_optimizer', anonymous=True)
        
        # 加載地圖
        self.load_slam_map('/home/chihsun/catkin_ws/src/my_robot_control/scripts/my_map0924.yaml')
        
        # 生成原始路徑點
        self.original_waypoints = self.generate_waypoints()
        
        # 創建KD樹
        self.kd_tree = self.build_obstacle_kdtree()
        
        # 初始化結果保存路徑
        self.result_folder = '/home/chihsun/catkin_ws/src/my_robot_control/scripts/optimized_path_img/'
        self.result_file_txt = '/home/chihsun/catkin_ws/src/my_robot_control/scripts/weight_test_results.txt'

    def load_slam_map(self, yaml_path):
        """加載SLAM地圖"""
        with open(yaml_path, 'r') as file:
            map_metadata = yaml.safe_load(file)
            self.map_origin = map_metadata['origin']
            self.map_resolution = map_metadata['resolution']
            
            # 讀取PNG地圖
            png_image = Image.open('/home/chihsun/catkin_ws/src/my_robot_control/scripts/my_map0924_4.png').convert('L')
            self.slam_map = np.array(png_image)

        self.generate_costmap()
        print("Map loaded successfully")

    def generate_costmap(self):
        """生成代價地圖"""
        wall_color = np.array([100, 100, 100])
        wall_color2 = np.array([120, 120, 120])
        
        img = cv2.cvtColor(self.slam_map, cv2.COLOR_GRAY2BGR)
        wall_mask = cv2.inRange(img, wall_color, wall_color2)
        
        self.cost_map = np.zeros_like(self.slam_map)
        
        inner_dilation = 2
        outer_dilation = 5
        
        inner_kernel = np.ones((inner_dilation * 2 + 1, inner_dilation * 2 + 1), np.uint8)
        inner_dilated = cv2.dilate(wall_mask, inner_kernel, iterations=1)
        
        outer_kernel = np.ones((outer_dilation * 2 + 1, outer_dilation * 2 + 1), np.uint8)
        outer_dilated = cv2.dilate(wall_mask, outer_kernel, iterations=1)
        
        outer_only = cv2.subtract(outer_dilated, inner_dilated)
        
        self.cost_map[wall_mask > 0] = 254
        self.cost_map[inner_dilated > 0] = 190
        self.cost_map[outer_only > 0] = 100

    def build_obstacle_kdtree(self):
        """構建障礙物KD樹"""
        obstacle_points = [
            (x, y) for y in range(self.slam_map.shape[0]) 
            for x in range(self.slam_map.shape[1])
            if self.slam_map[y, x] < 250
        ]
        if not obstacle_points:
            print("Warning: No obstacles detected in map.")
            obstacle_points = [(0, 0)]
        return KDTree(obstacle_points)

    def gazebo_to_image_coords(self, gazebo_x, gazebo_y):
        """轉換Gazebo座標到圖像座標"""
        img_x = 2000 + gazebo_x * 20
        img_y = 2000 - gazebo_y * 20
        return int(img_x), int(img_y)

    def image_to_gazebo_coords(self, img_x, img_y):
        """轉換圖像座標到Gazebo座標"""
        gazebo_x = (img_x - 2000) / 20.0
        gazebo_y = (2000 - img_y) / 20.0
        return gazebo_x, gazebo_y

    def calculate_smoothness_cost(self, prev_prev_point, prev_point, current_point):
        """計算路徑平滑度代價"""
        delta_xi = np.array([prev_point[0] - prev_prev_point[0], 
                            prev_point[1] - prev_prev_point[1]])
        delta_xi1 = np.array([current_point[0] - prev_point[0], 
                            current_point[1] - prev_point[1]])
        
        norm_delta_xi = np.linalg.norm(delta_xi)
        norm_delta_xi1 = np.linalg.norm(delta_xi1)
        
        if norm_delta_xi < 1e-6 or norm_delta_xi1 < 1e-6:
            return 0.0
                
        delta_xi_normalized = delta_xi / norm_delta_xi
        delta_xi1_normalized = delta_xi1 / norm_delta_xi1
        
        angle = np.arctan2(
            np.cross(delta_xi_normalized, delta_xi1_normalized), 
            np.dot(delta_xi_normalized, delta_xi1_normalized)
        )
        
        angle = np.abs(angle)
        angle_degrees = np.degrees(angle)
        normalized_cost = angle_degrees / 30.0
        
        return normalized_cost

    def check_line_for_obstacles(self, start, end, num_points=20):
        """檢查兩點之間是否有障礙物"""
        x1, y1 = start
        x2, y2 = end

        x_vals = np.linspace(x1, x2, num_points)
        y_vals = np.linspace(y1, y2, num_points)

        for x, y in zip(x_vals, y_vals):
            x, y = int(round(x)), int(round(y))
            if not (0 <= x < self.cost_map.shape[1] and 0 <= y < self.cost_map.shape[0]):
                continue
            if self.cost_map[y, x] >= 254:
                return True
        return False

    def generate_waypoints(self):
        """生成原始路徑點"""
        # 這裡放入你的原始路徑點
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
            (-7.2213, -1.7003)
]
        return waypoints

    def optimize_path(self, obs_weight, fps_weight, grid_size=35):
        """使用給定權重優化路徑"""
        optimized_waypoints = []
        
        for i in range(len(self.original_waypoints) - 1):
            start_point = self.original_waypoints[i]
            goal_point = self.original_waypoints[i + 1]
            
            optimized_point = self.optimize_single_point(
                start_point, goal_point, 
                obs_weight, fps_weight, 
                optimized_waypoints,
                grid_size
            )
            optimized_waypoints.append(optimized_point)
        
        # 添加終點
        optimized_waypoints.append(self.original_waypoints[-1])
        return optimized_waypoints

    def optimize_single_point(self, start_point, goal_point, obs_weight, fps_weight, 
                            current_optimized_points, grid_size):
        """優化單個路徑點"""
        img_start_x, img_start_y = self.gazebo_to_image_coords(*start_point)
        img_goal_x, img_goal_y = self.gazebo_to_image_coords(*goal_point)

        # 計算搜索範圍內的最大障礙物距離
        max_obstacle_distance = 0
        for x in range(img_start_x - grid_size // 2, img_start_x + grid_size // 2):
            for y in range(img_start_y - grid_size // 2, img_start_y + grid_size // 2):
                if not (0 <= x < self.slam_map.shape[1] and 0 <= y < self.slam_map.shape[0]):
                    continue
                distance, _ = self.kd_tree.query((x, y))
                max_obstacle_distance = max(max_obstacle_distance, distance)

        # 初始化最佳點
        best_f_score = float('inf')
        best_point = (img_start_x, img_start_y)

        # 在網格中搜索最佳點
        for x in range(img_start_x - grid_size // 2, img_start_x + grid_size // 2):
            for y in range(img_start_y - grid_size // 2, img_start_y + grid_size // 2):
                if not (0 <= x < self.slam_map.shape[1] and 0 <= y < self.slam_map.shape[0]):
                    continue

                # 計算各種代價
                distance_to_goal = np.sqrt((x - img_goal_x) ** 2 + (y - img_goal_y) ** 2)
                obstacle_distance, _ = self.kd_tree.query((x, y))
                normalized_obstacle_distance = obstacle_distance / max_obstacle_distance
                
                # 計算平滑度代價
                smoothness_cost = 0
                if len(current_optimized_points) >= 2:
                    prev_prev_point = current_optimized_points[-2]
                    prev_point = current_optimized_points[-1]
                    current_point = self.image_to_gazebo_coords(x, y)
                    smoothness_cost = self.calculate_smoothness_cost(
                        prev_prev_point, prev_point, current_point
                    )

                # 計算總代價
                f_score = distance_to_goal * 0.1 + \
                         (-normalized_obstacle_distance * obs_weight) + \
                         (smoothness_cost * fps_weight)

                if f_score < best_f_score:
                    best_f_score = f_score
                    best_point = (x, y)

        # 轉換回Gazebo座標
        optimized_x, optimized_y = self.image_to_gazebo_coords(*best_point)
        return (optimized_x, optimized_y)

    def visualize_path(self, waypoints, obs_weight, fps_weight):
        """視覺化路徑"""
        # 創建RGB圖像
        cost_map_rgb = np.zeros((self.cost_map.shape[0], self.cost_map.shape[1], 3), dtype=np.uint8)
        cost_map_rgb[self.cost_map == 0] = [255, 255, 255]
        cost_map_rgb[self.cost_map == 100] = [200, 200, 255]
        cost_map_rgb[self.cost_map == 190] = [150, 150, 255]
        cost_map_rgb[self.cost_map == 254] = [100, 100, 100]

        # 轉換路徑點到圖像座標
        img_points = [self.gazebo_to_image_coords(p[0], p[1]) for p in waypoints]

        plt.figure(figsize=(12, 12))
        plt.imshow(cost_map_rgb)

        # 繪製路徑點
        for point in img_points:
            plt.scatter(point[0], point[1], color='green', s=3)

        # 標記起點和終點
        plt.scatter(img_points[0][0], img_points[0][1], color='blue', s=100, marker='^', label='Start')
        plt.scatter(img_points[-1][0], img_points[-1][1], color='red', s=100, marker='v', label='Goal')

        plt.legend()
        plt.title(f'Path with obs_weight={obs_weight}, fps_weight={fps_weight}')
        
        # 保存圖片
        save_path = f'{self.result_folder}path_obs{obs_weight}_fps{fps_weight}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def run_weight_tests(self):
        """運行權重測試"""
        # 清空結果文件
        with open(self.result_file_txt, 'w') as f:
            f.write("Path Points with Different Weights\n")

        # 測試不同權重組合
        for obs_weight in range(1, 11):
            for fps_weight in range(1, 11):
                print(f"Testing weights: obs={obs_weight}, fps={fps_weight}")
                
                # 優化路徑
                optimized_path = self.optimize_path(obs_weight, fps_weight)
                
                # 保存路徑點
                with open(self.result_file_txt, 'a') as f:
                    f.write(f"\nweight_obs: {obs_weight} * weight_fps: {fps_weight}\n")
                    path_points = [f"({x:.4f},{y:.4f})" for x, y in optimized_path]
                    f.write(", ".join(path_points) + "\n")
                
                # 視覺化並保存圖片
                self.visualize_path(optimized_path, obs_weight, fps_weight)

def main():
    optimizer = PathOptimizer()
    optimizer.run_weight_tests()

if __name__ == '__main__':
    main()