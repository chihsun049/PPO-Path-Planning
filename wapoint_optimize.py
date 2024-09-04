import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import open3d as o3d
import cv2

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型架构
class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        
        self._to_linear = None
        self._to_linear_size(observation_space)
        
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.fc2 = nn.Linear(128, 128)
        
        self.actor = nn.Linear(128, action_space)
        self.critic = nn.Linear(128, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_space))

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def _to_linear_size(self, observation_space):
        x = torch.zeros(1, *observation_space)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        self._to_linear = x.view(1, -1).size(1)
    
    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)
        
        print(f"Input to conv1: {x}")
        x = torch.relu(self.conv1(x))
        print(f"After conv1: {x}")
        x = torch.relu(self.conv2(x))
        print(f"After conv2: {x}")
        
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        print(f"After fc1: {x}")
        
        x = self.dropout(x)
        x, _ = self.lstm(x.unsqueeze(1))
        print(f"After lstm: {x}")
        
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        print(f"After fc3: {x}")
        
        action_mean = self.actor(x)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        print(f"Action mean and std: {action_mean} {action_std}")

        optimized_path = self.path_optimizer(x)
        value = self.critic(x)

        return action_mean, action_std, value, optimized_path

# 动态生成占用网格
def generate_dynamic_occupancy_grid(pcd, current_wp, grid_size=0.05, map_size=10):
    points = np.asarray(pcd.points)
    
    # 检查点云数据是否有效
    if len(points) == 0:
        print("Warning: Empty point cloud data.")
        return np.zeros((4, 64, 64))  # 返回空的占用栅格

    if np.isnan(points).any() or np.isinf(points).any():
        print("Warning: Invalid point cloud data detected.")
        points = np.nan_to_num(points, nan=0.0, posinf=100.0, neginf=-100.0)

    cx, cy, yaw = current_wp
    map_size_in_grids = int(map_size / grid_size)
    grid = np.zeros((map_size_in_grids, map_size_in_grids))

    x_min, x_max = cx - map_size / 2, cx + map_size / 2
    y_min, y_max = cy - map_size / 2, cy + map_size / 2

    for point in points:
        x = point[0]
        y = point[1]

        if x_min <= x <= x_max and y_min <= y <= y_max:
            ix = int((x - x_min) / grid_size)
            iy = int((y - y_min) / grid_size)

            if 0 <= ix < grid.shape[1] and 0 <= iy < grid.shape[0]:
                grid[iy, ix] = 1

    grid = cv2.resize(grid, (64, 64), interpolation=cv2.INTER_LINEAR)
    
    waypoint_grid = np.zeros((3, 64, 64))
    waypoint_grid[0, :, :] = cx
    waypoint_grid[1, :, :] = cy
    waypoint_grid[2, :, :] = yaw
    
    occupancy_grid = np.vstack([grid[np.newaxis, :, :], waypoint_grid])
    
    return occupancy_grid

# 加载预训练模型
model = ActorCritic((4, 64, 64), 2).to(device)
model.load_state_dict(torch.load("/home/chihsun/catkin_ws/src/my_robot_control/scripts/best_model.pth", map_location=device))
model.eval()

# 加载CSV数据
input_csv_path = "/home/chihsun/shared_dir/0822-1floor/saved_waypoints.csv"
waypoints = pd.read_csv(input_csv_path)

# 加载点云文件
pcd_path = "/home/chihsun/shared_dir/0822-1floor/autoware-240822.pcd"
pcd = o3d.io.read_point_cloud(pcd_path)

# 优化路径
optimized_waypoints = []
for _, row in waypoints.iterrows():
    current_wp = np.array([row['x'], row['y'], row['yaw']])

    occupancy_grid = generate_dynamic_occupancy_grid(pcd, current_wp)

    # 检查输入数据是否正常
    print(f"Occupancy grid before input to the model: {occupancy_grid}")
    if np.isnan(occupancy_grid).any() or np.isinf(occupancy_grid).any():
        print(f"Invalid occupancy grid detected: {occupancy_grid}")
        occupancy_grid = np.nan_to_num(occupancy_grid, nan=0.0, posinf=0.0, neginf=0.0)

    state = torch.tensor(occupancy_grid, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, _, optimized_coordinates = model(state)
        optimized_coordinates = optimized_coordinates.cpu().data.numpy().flatten()
        print(f"Optimized coordinates: {optimized_coordinates}")

        # 检查是否生成了无效的坐标
        if np.isnan(optimized_coordinates).any() or np.isinf(optimized_coordinates).any():
            print(f"Invalid optimized coordinates detected: {optimized_coordinates}")
            optimized_coordinates = np.nan_to_num(optimized_coordinates, nan=0.0, posinf=0.0, neginf=0.0)

    optimized_row = row.copy()
    optimized_row['x'] = optimized_coordinates[0]
    optimized_row['y'] = optimized_coordinates[1]
    optimized_row['yaw'] = optimized_coordinates[2]
    
    optimized_waypoints.append(optimized_row)

# 保存优化后的路径点
output_csv_path = "/home/chihsun/shared_dir/0822-1floor/optimized_waypoints.csv"
optimized_df = pd.DataFrame(optimized_waypoints)
optimized_df.to_csv(output_csv_path, index=False)

print(f"优化后的路径已保存到 {output_csv_path}")
