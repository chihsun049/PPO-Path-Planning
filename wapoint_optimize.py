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

# ActorCritic 类定义
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

# 载入模型
model_path = "/home/chihsun/catkin_ws/src/my_robot_control/scripts/best_model.pth"
observation_space = (3, 64, 64)
action_space = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ActorCritic(observation_space, action_space).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

def create_empty_pointcloud2():
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'map'  # 根據你的需求調整這個 frame_id

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]

    data = np.zeros(3, dtype=np.float32).tobytes()  # 一個空的點，所有坐標為 0

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

def filter_pointcloud(pointcloud_msg, voxel_size=0.1, ground_threshold=-1.5):
    points = []
    for point in pc2.read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True):
        if point[2] > ground_threshold:  # 過濾地面點
            points.append([point[0], point[1], point[2]])
    
    # 使用 Open3D 將數據轉換為點雲物件
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 應用體素網格濾波器進行下采樣
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd

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
        lidar_data_list.append(create_empty_pointcloud2())

    return lidar_data_list

def save_lidar_to_pcd(points, filename="output.pcd"):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, point_cloud)
    print(f"Point cloud saved to {filename}")

def load_obstacles_from_pcd(pcd_file):
    point_cloud = o3d.io.read_point_cloud(pcd_file)
    print(f"Loaded obstacle point cloud with {len(point_cloud.points)} points.")
    return point_cloud

def generate_optimized_waypoints(model, initial_waypoints, lidar_data_list, obstacle_map, threshold=0.5):
    optimized_waypoints = []
    state = initial_waypoints
    for waypoint in initial_waypoints:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            if len(state_tensor.shape) == 3:
                state_tensor = state_tensor.unsqueeze(1)

            state_tensor = state_tensor.repeat(1, 3, 1, 1)
            state_tensor = torch.nn.functional.interpolate(state_tensor, size=(64, 64))

            action, yaw = model.act(state_tensor)  # 同时获取 action 和 yaw
            action = action.cpu().numpy().squeeze()
            yaw = yaw.cpu().numpy().squeeze()

            # 更新 x, y 和 yaw 坐标
            new_x = waypoint[0] + action[0]
            new_y = waypoint[1] + action[1]
            new_yaw = waypoint[2] + yaw  # 根据模型输出更新 yaw

            next_waypoint = [new_x, new_y, new_yaw]

            lidar_data = lidar_data_list[-1]  # 根据需求选择要使用的点云数据
            if is_safe(next_waypoint, lidar_data, obstacle_map, threshold):
                optimized_waypoints.append(next_waypoint)
            else:
                optimized_waypoints.append(waypoint)
            state = next_waypoint
    return optimized_waypoints

def is_safe(waypoint, lidar_data, obstacle_map, threshold=0.5):
    x, y = waypoint[:2]
    min_distance = float('inf')
    if isinstance(lidar_data, o3d.geometry.PointCloud):
        for point in np.asarray(lidar_data.points):
            distance = np.linalg.norm([point[0] - x, point[1] - y])
            if distance < min_distance:
                min_distance = distance
    return min_distance > threshold

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

def save_to_csv(original_waypoints, optimized_waypoints, other_columns, header, output_csv):
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for original, optimized, other_data in zip(original_waypoints, optimized_waypoints, other_columns):
            writer.writerow(optimized + other_data)
        if len(original_waypoints) > len(optimized_waypoints):
            for remaining, other_data in zip(original_waypoints[len(optimized_waypoints):], other_columns[len(optimized_waypoints):]):
                writer.writerow(remaining + other_data)

def publish_tf():
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        translation = (0.0, 0.0, 0.0)
        rotation = tf.transformations.quaternion_from_euler(0, 0, 0)
        br.sendTransform(translation, rotation, rospy.Time.now(), "velodyne", "map")
        rate.sleep()

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

    optimized_waypoints = generate_optimized_waypoints(model, waypoints, lidar_data, obstacle_map, threshold=0.5)

    output_csv = '/home/chihsun/shared_dir/0822-1floor/optimized_waypoints.csv'
    save_to_csv(waypoints, optimized_waypoints, other_columns, header, output_csv)
    print(f"Optimized waypoints saved to {output_csv}")

    tf_thread.join()

if __name__ == "__main__":
    main()
