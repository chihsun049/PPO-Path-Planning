#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_path_optimization(csv_path):
    # 讀取 CSV 檔案
    df = pd.read_csv(csv_path)
    
    # 設定中文字體支援
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 創建一個包含多個子圖的圖表
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 距離權重對路徑長度的影響
    ax1 = fig.add_subplot(331)
    for g in df['g_weight'].unique():
        for s in df['smoothness_weight'].unique():
            mask = (df['g_weight'] == g) & (df['smoothness_weight'] == s)
            ax1.plot(df[mask]['distance_penalty_weight'], 
                    df[mask]['total_length'], 
                    marker='o',
                    label=f'g={g}, s={s}')
    ax1.set_xlabel('Distance Penalty Weight')
    ax1.set_ylabel('Total Length')
    ax1.set_title('距離權重對路徑長度的影響')
    ax1.legend()
    
    # 2. 平滑度與障礙物距離的關係散點圖
    ax2 = fig.add_subplot(332)
    scatter = ax2.scatter(df['avg_obstacle_distance'], 
                         df['smoothness'],
                         c=df['distance_penalty_weight'],
                         cmap='viridis')
    plt.colorbar(scatter, label='Distance Penalty Weight')
    ax2.set_xlabel('Average Obstacle Distance')
    ax2.set_ylabel('Smoothness')
    ax2.set_title('平滑度與障礙物距離的關係')
    
    # 3. 各參數對路徑長度的箱型圖
    ax3 = fig.add_subplot(333)
    sns.boxplot(data=df, y='total_length', x='distance_penalty_weight', ax=ax3)
    ax3.set_title('不同距離權重的路徑長度分布')
    
    # 4. 相關性熱圖
    ax4 = fig.add_subplot(334)
    correlation = df[['total_length', 'avg_obstacle_distance', 'smoothness']].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax4)
    ax4.set_title('指標間的相關性')
    
    # 5. 平滑度趨勢
    ax5 = fig.add_subplot(335)
    for g in df['g_weight'].unique():
        for s in df['smoothness_weight'].unique():
            mask = (df['g_weight'] == g) & (df['smoothness_weight'] == s)
            ax5.plot(df[mask]['distance_penalty_weight'], 
                    df[mask]['smoothness'], 
                    marker='o',
                    label=f'g={g}, s={s}')
    ax5.set_xlabel('Distance Penalty Weight')
    ax5.set_ylabel('Smoothness')
    ax5.set_title('距離權重對平滑度的影響')
    ax5.legend()
    
    # 6. 綜合性能評估
    ax6 = fig.add_subplot(336)
    # 標準化數據以便比較
    df_normalized = df.copy()
    for column in ['total_length', 'avg_obstacle_distance', 'smoothness']:
        df_normalized[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    
    df_normalized.plot(y=['total_length', 'avg_obstacle_distance', 'smoothness'], 
                      ax=ax6, marker='o')
    ax6.set_xlabel('測試序號')
    ax6.set_ylabel('標準化值')
    ax6.set_title('各指標的綜合比較(標準化)')
    
    # 7. 顯示平滑度與障礙物距離最佳組合
    ax7 = fig.add_subplot(337)
    smoothness_threshold = df['smoothness'].quantile(0.3)  # 前 10% 最小值
    obstacle_distance_threshold = df['avg_obstacle_distance'].quantile(0.7)  # 前 10% 最大值
    
    best_combinations = df[(df['smoothness'] <= smoothness_threshold) &
                           (df['avg_obstacle_distance'] >= obstacle_distance_threshold)]
    
    for _, row in best_combinations.iterrows():
        ax7.scatter(row['distance_penalty_weight'], 
                    row['smoothness'], 
                    color='red', 
                    label=f"g={row['g_weight']}, s={row['smoothness_weight']}, dpw={row['distance_penalty_weight']}")
    
    ax7.set_xlabel('Distance Penalty Weight')
    ax7.set_ylabel('Smoothness')
    ax7.set_title('最佳平滑與障礙物距離的權重組合')
    ax7.legend()
    
    # 調整布局
    plt.tight_layout()
    
    # 保存圖表
    plt.savefig('path_optimization_analysis_with_ax7.png', dpi=300, bbox_inches='tight')
    
    # 輸出統計摘要
    print("\n=== 統計摘要 ===")
    print("\n基本統計量：")
    print(df[['total_length', 'avg_obstacle_distance', 'smoothness']].describe())
    
    print("\n最佳參數組合：")
    best_length = df.loc[df['total_length'].idxmin()]
    best_smoothness = df.loc[df['smoothness'].idxmin()]
    best_distance = df.loc[df['avg_obstacle_distance'].idxmax()]
    
    print("\n最短路徑的參數組合:")
    print(f"g_weight: {best_length['g_weight']}")
    print(f"smoothness_weight: {best_length['smoothness_weight']}")
    print(f"distance_penalty_weight: {best_length['distance_penalty_weight']}")
    
    print("\n最平滑路徑的參數組合:")
    print(f"g_weight: {best_smoothness['g_weight']}")
    print(f"smoothness_weight: {best_smoothness['smoothness_weight']}")
    print(f"distance_penalty_weight: {best_smoothness['distance_penalty_weight']}")
    
    print("\n與障礙物距離最遠的參數組合:")
    print(f"g_weight: {best_distance['g_weight']}")
    print(f"smoothness_weight: {best_distance['smoothness_weight']}")
    print(f"distance_penalty_weight: {best_distance['distance_penalty_weight']}")
    
    print("\n最佳平滑與障礙物距離組合:")
    print(best_combinations[['g_weight', 'smoothness_weight', 'distance_penalty_weight', 
                              'total_length', 'avg_obstacle_distance', 'smoothness']])
    return best_combinations

# 使用分析函數
if __name__ == "__main__":
    analyze_path_optimization('/home/chihsun/catkin_ws/src/my_robot_control/scripts/optimized_path_img/final_report.csv')  # 替換為您的文件路徑
