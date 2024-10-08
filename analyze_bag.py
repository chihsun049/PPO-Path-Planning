import rosbag
from collections import defaultdict

def analyze_bag(bag_file):
    # 打開bag文件
    bag = rosbag.Bag(bag_file)
    
    # 創建字典來儲存話題的消息類型和計數
    topic_info = defaultdict(lambda: {"type": None, "count": 0})
    
    # 迭代所有的消息
    for topic, msg, t in bag.read_messages():
        if topic_info[topic]["type"] is None:
            topic_info[topic]["type"] = msg._type  # 設置消息類型
        topic_info[topic]["count"] += 1  # 增加消息計數
    
    # 打印話題信息
    print("Bag file analysis:")
    for topic, info in topic_info.items():
        print(f"Topic: {topic}")
        print(f"  Message Type: {info['type']}")
        print(f"  Message Count: {info['count']}")
        print()
    
    # 關閉bag文件
    bag.close()

# 設置要分析的bag文件路徑
bag_file = "/home/chihsun/shared_dir/0822-1floor/autoware-20240822064852.bag"
analyze_bag(bag_file)
