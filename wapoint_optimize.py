import torch
import csv
import rospy
import rosbag
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import torch.nn as nn
import open3d as o3d
import numpy as np
import threading
import tf
import std_msgs.msg
from scipy.special import comb

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ActorCritic 類定義
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

# 平滑函數
def smooth_waypoints(waypoints, combined_map, threshold=0.5, n_points=100):
    waypoints = np.array(waypoints)
    n = len(waypoints) - 1

    def bernstein_poly(i, n, t):
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

    t = np.linspace(0.0, 1.0, n_points)
    curve = np.zeros((n_points, 2))

    for i in range(n + 1):
        curve += np.outer(bernstein_poly(i, n, t), waypoints[i][:2])

    # 檢查並保存平滑後的點
    smoothed_waypoints = []
    for idx, pt in enumerate(curve):
        smoothed_point = (pt[0], pt[1], 0)
        if is_safe(smoothed_point, combined_map, threshold):
            smoothed_waypoints.append(smoothed_point)
        else:
            # 退回到最近的原始路徑點，避免生成重複點
            smoothed_waypoints.append(waypoints[idx % len(waypoints)])
    
    return smoothed_waypoints

def convert_pointcloud2_to_open3d(lidar_data):
    # 轉換 PointCloud2 為 numpy 數據
    points = []
    for point in pc2.read_points(lidar_data, field_names=("x", "y", "z"), skip_nans=True):
        points.append([point[0], point[1], point[2]])
    
    # 創建 open3d 點雲物件
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float32))
    return pcd

def process_lidar_and_pcd(lidar_data, obstacle_map, voxel_size=0.1):
    # 轉換 PointCloud2 格式為 Open3D 點雲格式
    lidar_pcd = convert_pointcloud2_to_open3d(lidar_data)
    
    # 結合 LiDAR 和障礙物地圖
    combined_map = lidar_pcd + obstacle_map
    combined_map = combined_map.voxel_down_sample(voxel_size)
    
    return combined_map

def is_safe(waypoint, combined_map, threshold=0.5):
    x, y = waypoint[:2]
    min_distance = float('inf')
    for point in np.asarray(combined_map.points):
        distance = np.linalg.norm([point[0] - x, point[1] - y])
        if distance < min_distance:
            min_distance = distance
    return min_distance > threshold

def generate_optimized_waypoints(model, initial_waypoints, lidar_data_list, obstacle_map, threshold=0.5, max_distance=0.5):
    optimized_waypoints = []
    state = initial_waypoints
    optimized_waypoints.append(initial_waypoints[0])  # 保留第一個點

    for waypoint in initial_waypoints[1:-1]:  # 略過第一個和最後一個點
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            
            if len(state_tensor.shape) == 3:
                state_tensor = state_tensor.unsqueeze(1)
                
            state_tensor = state_tensor.repeat(1, 3, 1, 1)
            state_tensor = torch.nn.functional.interpolate(state_tensor, size=(64, 64))
            
            action, yaw = model.act(state_tensor)
            action = action.cpu().numpy().squeeze()
            yaw = yaw.cpu().numpy().squeeze()
            
            new_x = waypoint[0] + action[0]
            new_y = waypoint[1] + action[1]
            new_yaw = waypoint[2] + yaw

            prev_x, prev_y, _ = optimized_waypoints[-1]
            distance = np.sqrt((new_x - prev_x) ** 2 + (new_y - prev_y) ** 2)
            
            if distance > max_distance:
                direction = np.arctan2(new_y - prev_y, new_x - prev_x)
                new_x = prev_x + max_distance * np.cos(direction)
                new_y = prev_y + max_distance * np.sin(direction)
            
            next_waypoint = [new_x, new_y, new_yaw]
            
            if lidar_data_list:
                lidar_data = lidar_data_list[-1]
                combined_map = process_lidar_and_pcd(lidar_data, obstacle_map)
                if is_safe(next_waypoint, combined_map, threshold):
                    optimized_waypoints.append(next_waypoint)
                else:
                    optimized_waypoints.append(waypoint)
            else:
                rospy.logwarn("No LiDAR data available, skipping obstacle check.")
                optimized_waypoints.append(waypoint)
            state = next_waypoint

    optimized_waypoints.append(initial_waypoints[-1])
    
    return optimized_waypoints

def create_empty_pointcloud2():
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'map'  # 根據需求調整這個 frame_id

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]

    # 空點資料，所有坐標設為 0
    data = np.zeros(3, dtype=np.float32).tobytes() 

    pointcloud2_msg = PointCloud2(
        header=header,
        height=1,
        width=1,
        is_dense=True,
        is_bigendian=False,
        fields=fields,
        point_step=12,
        row_step=12,
        data=data
    )
    
    return pointcloud2_msg

# 讀取LiDAR數據
def load_lidar_from_bag(bag_file, topic="/points_raw"):
    try:
        bag = rosbag.Bag(bag_file)
    except Exception as e:
        rospy.logerr(f"Failed to open bag file: {e}")
        return []

    lidar_data_list = []
    for topic_name, msg, t in bag.read_messages(topics=[topic]):
        rospy.loginfo(f"Reading message from topic {topic_name}")
        if isinstance(msg, PointCloud2):
            rospy.loginfo(f"Found PointCloud2 data at time {t.to_sec()}")
            filtered_pcd = filter_pointcloud(msg)  # 對數據進行濾波
            lidar_data_list.append(filtered_pcd)
    bag.close()

    rospy.loginfo(f"Collected {len(lidar_data_list)} filtered LiDAR data messages from {topic}")
    
    if not lidar_data_list:
        rospy.logwarn(f"No LiDAR data found in the bag file on topic {topic}. Using empty PointCloud2.")
        lidar_data_list.append(create_empty_pointcloud2())  # 避免空數據返回

    return lidar_data_list

def filter_pointcloud(pointcloud_msg, voxel_size=0.1, ground_threshold=-1.5):
    points = []
    for point in pc2.read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True):
        if point[2] > ground_threshold:  # 過濾地面點
            points.append([point[0], point[1], point[2]])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd

# 讀取障礙物點雲數據
def load_obstacles_from_pcd(pcd_file):
    point_cloud = o3d.io.read_point_cloud(pcd_file)
    print(f"Loaded obstacle point cloud with {len(point_cloud.points)} points.")
    return point_cloud

# 讀取CSV路徑點
def load_csv_waypoints(csv_file):
    waypoints = []
    other_columns = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            if len(row) >= 3:
                x, y, yaw = map(float, row[:3])
                waypoints.append([x, y, yaw])
                other_columns.append(row[3:])
            else:
                rospy.logwarn(f"Skipping invalid row: {row}")
    return header, waypoints, other_columns

# 儲存優化後的路徑點至CSV
def save_to_csv(original_waypoints, optimized_waypoints, other_columns, header, output_csv):
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for original, optimized, other_data in zip(original_waypoints, optimized_waypoints, other_columns):
            # 將 optimized 轉換為 list，然後再進行連接
            writer.writerow(list(optimized) + other_data)
        if len(original_waypoints) > len(optimized_waypoints):
            for remaining, other_data in zip(original_waypoints[len(optimized_waypoints):], other_columns[len(optimized_waypoints):]):
                writer.writerow(remaining + other_data)

# TF 廣播函數
def publish_tf():
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        try:
            translation = (0.0, 0.0, 0.0)
            rotation = tf.transformations.quaternion_from_euler(0, 0, 0)
            br.sendTransform(translation, rotation, rospy.Time.now(), "velodyne", "map")
            rate.sleep()
        except rospy.ROSInterruptException:
            rospy.loginfo("ROS shutdown request detected. Exiting publish_tf.")
            break

# 主程式入口
def main():
    rospy.init_node('waypoint_optimizer', anonymous=True)
    rospy.loginfo("Starting TF broadcaster...")
    tf_thread = threading.Thread(target=publish_tf)
    tf_thread.daemon = True
    tf_thread.start()

    bag_file = "/home/chihsun/shared_dir/0822-1floor/autoware-20240822064852.bag"
    lidar_data = load_lidar_from_bag(bag_file, topic="/points_raw")

    pcd_file = "/home/chihsun/shared_dir/0822-1floor/autoware-240822.pcd"
    obstacle_map = load_obstacles_from_pcd(pcd_file)

    csv_file = '/home/chihsun/shared_dir/0822-1floor/saved_waypoints.csv'
    header, waypoints, other_columns = load_csv_waypoints(csv_file)

    model_path = "/home/chihsun/catkin_ws/src/my_robot_control/scripts/saved_model_ppo.pth"
    observation_space = (3, 64, 64)
    action_space = 2
    model = ActorCritic(observation_space, action_space).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    optimized_waypoints = generate_optimized_waypoints(model, waypoints, lidar_data, obstacle_map, threshold=0.5)

    # 結合 LiDAR 和障礙物地圖
    if lidar_data:
        combined_map = process_lidar_and_pcd(lidar_data[-1], obstacle_map)
    else:
        combined_map = obstacle_map

    # 平滑優化後的路徑點
    smoothed_waypoints = smooth_waypoints(optimized_waypoints, combined_map, threshold=0.5)

    output_csv = '/home/chihsun/shared_dir/0822-1floor/optimized_waypoints.csv'
    save_to_csv(waypoints, smoothed_waypoints, other_columns, header, output_csv)
    print(f"Optimized waypoints saved to {output_csv}")

    tf_thread.join()

if __name__ == "__main__":
    main()
