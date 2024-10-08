import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
csv_file = '/home/chihsun/shared_dir/0822-1floor/optimized_waypoints.csv'
#csv_file = '/home/chihsun/shared_dir/0822-1floor/saved_waypoints.csv'
df = pd.read_csv(csv_file)

# 初始化图像
fig, ax = plt.subplots()
ax.plot(df['x'], df['y'], 'bo-', label='Path')  # 路径和蓝点
ax.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Vehicle Path Visualization')

# 初始化点列表
points = df[['x', 'y', 'velocity', 'yaw']].values.tolist()

# 显示每个点的速度和方向箭头
arrow_length = 0.1  # 箭头长度，可以根据需要调整
for index, row in df.iterrows():
    ax.arrow(row['x'], row['y'], arrow_length * np.cos(row['yaw']), arrow_length * np.sin(row['yaw']), 
             head_width=0.2, head_length=0.3, fc='red', ec='red')  # 增大箭头尺寸

# 标记起始点和终点
if not df.empty:
    ax.plot(df['x'].iloc[0], df['y'].iloc[0], 'go', label='Start')  # 起始点为绿色
    ax.plot(df['x'].iloc[-1], df['y'].iloc[-1], 'ro', label='End')  # 终点为红色
    ax.legend()

# 清除所有注释的函数
def clear_annotations():
    for artist in ax.texts + ax.patches:
        if isinstance(artist, plt.Text):
            artist.remove()

# 鼠标移动事件的处理函数
def onmove(event):
    if event.inaxes is not None:
        clear_annotations()
        for i, point in enumerate(points):
            if np.isclose(event.xdata, point[0], atol=0.1) and np.isclose(event.ydata, point[1], atol=0.1):
                # 显示该点的数据
                annotation_text = (f"Point: {i}\n"
                                   f"x: {point[0]:.2f}, y: {point[1]:.2f}\n"
                                   f"velocity: {point[2]:.2f}, yaw: {point[3]:.2f}")
                ax.annotate(annotation_text, (point[0], point[1]), textcoords="offset points", 
                            xytext=(10,10), ha='center', color='red', 
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
                break
        fig.canvas.draw()

# 点击事件的处理函数
selected_index = None

def onclick(event):
    global selected_index
    if event.inaxes is not None:
        for i, point in enumerate(points):
            if np.isclose(event.xdata, point[0], atol=0.1) and np.isclose(event.ydata, point[1], atol=0.1):
                selected_index = i
                break

# 拖动事件的处理函数
def ondrag(event):
    if selected_index is not None and event.inaxes is not None:
        points[selected_index][0] = event.xdata
        points[selected_index][1] = event.ydata
        df.loc[selected_index, 'x'] = event.xdata
        df.loc[selected_index, 'y'] = event.ydata
        ax.clear()
        ax.plot(df['x'], df['y'], 'bo-', label='Path')
        for index, row in df.iterrows():
            ax.arrow(row['x'], row['y'], arrow_length * np.cos(row['yaw']), arrow_length * np.sin(row['yaw']), 
                     head_width=0.2, head_length=0.3, fc='red', ec='red')
        if not df.empty:
            ax.plot(df['x'].iloc[0], df['y'].iloc[0], 'go', label='Start')
            ax.plot(df['x'].iloc[-1], df['y'].iloc[-1], 'ro', label='End')
            ax.legend()
        fig.canvas.draw()

# 释放鼠标事件的处理函数
def onrelease(event):
    global selected_index
    selected_index = None

# 保存函数
def save_path(event=None):
    new_df = pd.DataFrame(points, columns=['x', 'y', 'velocity', 'yaw'])
    excel_file = '/home/chihsun/shared_dir/0723/smoothed_path.xlsx'
    new_df.to_excel(excel_file, index=False)
    print(f"Path saved to {excel_file}")

# 连接事件处理函数
fig.canvas.mpl_connect('motion_notify_event', onmove)
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('motion_notify_event', ondrag)
fig.canvas.mpl_connect('button_release_event', onrelease)

# 关闭窗口事件的处理函数
def on_close(event):
    save_path()

fig.canvas.mpl_connect('close_event', on_close)

plt.show()
