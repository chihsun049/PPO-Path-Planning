import torch
import csv
import rospy
import rosbag
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import torch.nn as nn
import torch.nn.functional as F
from open3d import io as o3d_io
import numpy as np
import open3d as o3d
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

        return action_mean, action_std, value

    def act(self, state):
        action_mean, action_std, _ = self(state)
        action = action_mean + action_std * torch.randn_like(action_std)
        action = torch.tanh(action)
        max_action = torch.tensor([2.0, 0.6], device=action.device)
        min_action = torch.tensor([-2.0, -0.6], device=action.device)
        action = min_action + (action + 1) * (max_action - min_action) / 2
        return action.detach()

    def evaluate(self, state, action):
        action_mean, action_std, value = self(state)
        dist = torch.distributions.Normal(action_mean, action_std)
        action_log_probs = dist.log_prob(action).sum(dim=-1, keepdim=True)
        dist_entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return action_log_probs, value, dist_entropy

# 载入模型
model_path = "/home/chihsun/catkin_ws/src/my_robot_control/scripts/saved_model_ppo.pth"
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

# 读取 LiDAR 数据的函数
def load_lidar_from_bag(bag_file, topic="/points_raw"):
    bag = rosbag.Bag(bag_file)
    lidar_data_list = []
    for topic_name, msg, t in bag.read_messages(topics=[topic]):
        if isinstance(msg, PointCloud2):
            print(f"LiDAR data found at time {t.to_sec()} on topic {topic_name}")
            lidar_data_list.append(msg)
    bag.close()

    if not lidar_data_list:
        print(f"Warning: No LiDAR data found in the bag file on topic {topic}. Using empty PointCloud2.")
        lidar_data_list.append(create_empty_pointcloud2())  # 使用虛擬的空數據

    return lidar_data_list

# 从 PCD 文件加载障碍物点云
def load_obstacles_from_pcd(pcd_file):
    point_cloud = o3d_io.read_point_cloud(pcd_file)
    print(f"Loaded obstacle point cloud with {len(point_cloud.points)} points.")
    return point_cloud

# 优化路径点
def generate_optimized_waypoints(model, initial_waypoints, lidar_data_list, obstacle_map):
    optimized_waypoints = []
    state = initial_waypoints
    for waypoint in initial_waypoints:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            if len(state_tensor.shape) == 3:
                state_tensor = state_tensor.unsqueeze(1)

            state_tensor = state_tensor.repeat(1, 3, 1, 1)
            state_tensor = torch.nn.functional.interpolate(state_tensor, size=(64, 64))

            action = model.act(state_tensor)
            action = action.cpu().numpy().squeeze()

            next_waypoint = [waypoint[0] + action[0], waypoint[1] + action[1], waypoint[2]]

            # 处理多个 LiDAR 数据，假设只取最后一个 LiDAR 数据进行判断
            lidar_data = lidar_data_list[-1]  # 你可以根据需求选择要使用的点云数据
            if is_safe(next_waypoint, lidar_data, obstacle_map):
                optimized_waypoints.append(next_waypoint)
            else:
                # 如果不安全，保留原始的路径点
                optimized_waypoints.append(waypoint)
            state = next_waypoint
    return optimized_waypoints

# 判断路径点是否安全
def is_safe(waypoint, lidar_data, obstacle_map, threshold=0.5):
    x, y = waypoint[:2]
    min_distance = float('inf')
    # 确保 lidar_data 是单个 PointCloud2 消息
    if isinstance(lidar_data, PointCloud2):
        # 遍历 LiDAR 数据点，判断是否靠近障碍物
        for point in pc2.read_points(lidar_data, field_names=("x", "y", "z"), skip_nans=True):
            distance = np.linalg.norm([point[0] - x, point[1] - y])
            if distance < min_distance:
                min_distance = distance
    # 判断路径点是否靠近障碍物
    return min_distance > threshold

# 从CSV文件读取初始路径点，包括标头
def load_csv_waypoints(csv_file):
    waypoints = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # 读取并保留标头
        for row in reader:
            if len(row) >= 3:
                x, y, yaw = map(float, row[:3])
                waypoints.append([x, y, yaw])
            else:
                rospy.logwarn(f"Skipping invalid row: {row}")
    return header, waypoints

# 保存优化后的路径点到CSV，保留未修改的路径和原始标头
def save_to_csv(original_waypoints, optimized_waypoints, header, output_csv):
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # 写入标头

        for original, optimized in zip(original_waypoints, optimized_waypoints):
            writer.writerow(optimized)  # 无论是否修改，都写入优化后的路径点

        # 如果原始路径点多于优化路径点，继续写入剩下的原始路径点
        if len(original_waypoints) > len(optimized_waypoints):
            for remaining in original_waypoints[len(optimized_waypoints):]:
                writer.writerow(remaining)

# TF 變換發佈器
def publish_tf():
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        translation = (0.0, 0.0, 0.0)
        rotation = tf.transformations.quaternion_from_euler(0, 0, 0)
        br.sendTransform(translation, rotation, rospy.Time.now(), "velodyne", "map")
        rate.sleep()

# 主函数修改
def main():
    rospy.init_node('waypoint_optimizer')

    # 开始 TF 变换的发布线程
    rospy.loginfo("Starting TF broadcaster...")
    tf_thread = threading.Thread(target=publish_tf)
    tf_thread.start()

    # 从 bag 文件读取 LiDAR 数据
    bag_file = "/home/chihsun/shared_dir/0822-1floor/autoware-20240822064852.bag"
    lidar_data = load_lidar_from_bag(bag_file, topic="/points_raw")

    # 从 PCD 文件加载障碍物点云
    pcd_file = "/home/chihsun/shared_dir/0822-1floor/autoware-240822.pcd"
    obstacle_map = load_obstacles_from_pcd(pcd_file)

    # 载入初始的路径点CSV文件和标头
    csv_file = '/home/chihsun/shared_dir/0822-1floor/saved_waypoints.csv'
    header, waypoints = load_csv_waypoints(csv_file)

    # 生成优化后的路径点
    optimized_waypoints = generate_optimized_waypoints(model, waypoints, lidar_data, obstacle_map)

    # 保存优化后的路径点到CSV，保留原始未修改的路径点和标头
    output_csv = '/home/chihsun/shared_dir/0822-1floor/optimized_waypoints.csv'
    save_to_csv(waypoints, optimized_waypoints, header, output_csv)
    print(f"Optimized waypoints saved to {output_csv}")

    # 停止 TF 發布线程
    tf_thread.join()

if __name__ == "__main__":
    main()