import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import folium
from folium.plugins import HeatMap
import os
import warnings
import matplotlib.ticker as mtick

# 1. 基础设置
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
sns.set_style("whitegrid", {'font.sans-serif': ['SimHei']}) # 设置seaborn风格

# 创建结果目录结构
os.makedirs('analysis_results2', exist_ok=True)
os.makedirs('analysis_results2/maps', exist_ok=True)
os.makedirs('analysis_results2/trends', exist_ok=True)

def analyze_taxi_data_monthly(file_path):
    print("="*50)
    print("网约车业务数据分析工具 (月度分析 | 含热门地点与时长分布)")
    print(f"分析文件: {file_path}")
    print("="*50)

    try:
        # ---------------------------------------------------------
        # [1/5] 数据加载与特征提取
        # ---------------------------------------------------------
        print("\n[1/5] 数据加载与预处理...")

        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='gbk')

        print(f"原始数据量: {len(df)} 条记录")

        # 转换时间格式
        df['dep_time'] = pd.to_datetime(df['dep_time'], errors='coerce')
        df['dest_time'] = pd.to_datetime(df['dest_time'], errors='coerce')
        df.dropna(subset=['dep_time', 'dest_time'], inplace=True)

        # === 时间特征 ===
        df['date'] = df['dep_time'].dt.date
        df['weekday'] = df['dep_time'].dt.dayofweek  # 0=周一
        df['hour'] = df['dep_time'].dt.hour

        # 映射中文星期，方便画图
        weekday_map = {0:'周一', 1:'周二', 2:'周三', 3:'周四', 4:'周五', 5:'周六', 6:'周日'}
        df['weekday_cn'] = df['weekday'].map(weekday_map)

        # 创建一个排序用的标签： "07-01(周一)"
        df['date_str'] = df['dep_time'].dt.strftime('%m-%d')
        df['day_label'] = df['date_str'] + '\n' + df['weekday_cn']

        # 计算时长 (分钟)并清洗异常值（保留1分钟到300分钟的正常订单）
        df['trip_duration'] = (df['dest_time'] - df['dep_time']).dt.total_seconds() / 60
        df = df[(df['trip_duration'] > 1) & (df['trip_duration'] < 300)]

        # ---------------------------------------------------------
        # [2/5] 月度趋势与时长分析
        # ---------------------------------------------------------
        print("\n[2/5] 生成月度趋势与时长图表...")

        # A. 每日订单量走势
        daily_stats = df.groupby(['date', 'day_label']).size().reset_index(name='count')
        daily_stats.sort_values('date', inplace=True)

        # 针对月度数据，拉长画布，并对 x 轴标签做旋转防止重叠
        plt.figure(figsize=(16, 6))
        ax = sns.barplot(x='day_label', y='count', data=daily_stats, palette="Blues_d", alpha=0.8)
        sns.lineplot(x='day_label', y='count', data=daily_stats, marker='o', color='red', linewidth=2, ax=ax)

        # 在柱子上标注具体数字 (为避免太挤，稍微调小字体，并旋转文字)
        for i, v in enumerate(daily_stats['count']):
            ax.text(i, v + 50, str(v), ha='center', va='bottom', fontsize=9, rotation=0)

        plt.title('月度每日订单量变化趋势', fontsize=15, pad=10)
        plt.xlabel('')
        plt.ylabel('订单量')
        plt.xticks(rotation=45) # 倾斜 45 度，完美展示 31 天
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('analysis_results2/trends/monthly_trend.png', dpi=300)
        plt.close()

        # B. 运营节奏热力图 (仅工作日)
        print("  - 生成运营节奏热力图 (月度工作日均值)...")

        # 先筛选出 0-4 (周一到周五) 的数据进行聚合
        heatmap_df = df[df['weekday'] <= 4]
        heatmap_data = heatmap_df.groupby(['hour', 'weekday']).size().reset_index(name='count')

        # 因为是月度数据，热力图的总数会很大。如果想看“单日平均”，可以除以该星期几在月内的天数
        # 但这里为了保持原本“总和”的功能不变，直接用 count 也可以，色彩分布趋势是一样的
        heatmap_pivot = heatmap_data.pivot(index='hour', columns='weekday', values='count')

        # 确保周一到周五列都存在
        for i in range(5):  # 只循环 0,1,2,3,4
            if i not in heatmap_pivot.columns:
                heatmap_pivot[i] = 0

        # 排序并重命名列
        heatmap_pivot = heatmap_pivot[sorted(heatmap_pivot.columns)]
        heatmap_pivot.columns = [weekday_map[i] for i in heatmap_pivot.columns]

        plt.figure(figsize=(10, 8))
        # 绘制热力图
        sns.heatmap(heatmap_pivot, cmap='YlGnBu', annot=False, fmt='d', linewidths=0.5)
        plt.title('运营节奏热力图 (月度·周一至周五累积)', fontsize=14)
        plt.ylabel('小时 (0-23)')
        plt.xlabel('工作日')
        plt.savefig('analysis_results2/trends/monthly_heatmap.png', dpi=300)
        plt.close()

        # C. 订单时间（行程时长）从长到短排序图 (功能保持不变)
        print("  - 生成订单行程时长降序分布图...")
        # 提取时长并降序排列
        sorted_durations = df['trip_duration'].sort_values(ascending=False).values
        total_orders_count = len(sorted_durations)

        # 计算30分钟以内的订单量和占比
        count_under_30 = np.sum(sorted_durations <= 30)
        pct_under_30 = count_under_30 / total_orders_count * 100
        x_pos_30 = 100 - pct_under_30 # 因为是降序，30分钟所在的位置是 100% 减去 <=30分的占比

        # 将横坐标转换为百分比比例 (0% - 100%)
        x_percentages = np.linspace(0, 100, total_orders_count)

        plt.figure(figsize=(12, 6))
        # 使用面积填充图使得分布曲线更直观
        plt.plot(x_percentages, sorted_durations, color='#2ca02c', linewidth=2)
        plt.fill_between(x_percentages, sorted_durations, color='#2ca02c', alpha=0.3)

        # 添加平均时长参考线
        avg_dur = df['trip_duration'].mean()
        plt.axhline(avg_dur, color='red', linestyle='--', linewidth=1.5, label=f'平均行程时长: {avg_dur:.1f} 分钟')

        # 标注30分钟界线与订单量
        plt.plot([x_pos_30, x_pos_30], [0, 30], color='darkorange', linestyle=':', linewidth=2)
        plt.plot([0, x_pos_30], [30, 30], color='darkorange', linestyle=':', linewidth=2)
        plt.scatter([x_pos_30], [30], color='darkorange', s=60, zorder=5, label='30分钟分界点')

        # 添加带箭头的文本标注
        text_x = x_pos_30 + 3 if x_pos_30 < 80 else x_pos_30 - 15
        plt.annotate(f'<=30分钟订单占比: {pct_under_30:.1f}%\n(共计 {count_under_30:,} 单)',
                     xy=(x_pos_30, 30),
                     xytext=(text_x, 50),
                     arrowprops=dict(facecolor='darkorange', edgecolor='darkorange', shrink=0.05, width=1.5, headwidth=7),
                     fontsize=11, color='darkorange', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="darkorange", alpha=0.8))

        # 优化坐标轴显示范围
        plt.xlim(0, 100)
        y_max = np.percentile(sorted_durations, 99.9)
        plt.ylim(0, y_max)

        # 格式化横坐标为百分比格式
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())

        plt.title('月度订单行程时长分布 (从长到短降序排列)', fontsize=14, pad=15)
        plt.xlabel('订单比例 (%)', fontsize=12)
        plt.ylabel('行程时长 (分钟)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(axis='both', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig('analysis_results2/trends/duration_descending.png', dpi=300)
        plt.close()

        # ---------------------------------------------------------
        # [3/5] 热门地点 Top 5 分析
        # ---------------------------------------------------------
        print("\n[3/5] 统计热门上下车地点...")

        top_dep_str = "            无数据"
        top_dest_str = "            无数据"

        # 检查是否存在地点名称列
        if 'dep_area' in df.columns and 'dest_area' in df.columns:
            # 统计 Top 5
            top5_dep = df['dep_area'].value_counts().head(5)
            top5_dest = df['dest_area'].value_counts().head(5)

            # 绘制 Top 5 横向柱状图
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))

            # 上车点图
            sns.barplot(x=top5_dep.values, y=top5_dep.index, ax=axes[0], palette='viridis')
            axes[0].set_title('月度 Top 5 热门上车点')
            axes[0].set_xlabel('订单量')

            # 下车点图
            sns.barplot(x=top5_dest.values, y=top5_dest.index, ax=axes[1], palette='magma')
            axes[1].set_title('月度 Top 5 热门下车点')
            axes[1].set_xlabel('订单量')

            plt.tight_layout()
            plt.savefig('analysis_results2/trends/top_locations.png', dpi=300)
            plt.close()

            # 生成报告用的字符串 (统一添加12个空格进行对齐)
            top_dep_str = "\n".join([f"            {i+1}. {name} ({count}单)" for i, (name, count) in enumerate(top5_dep.items())])
            top_dest_str = "\n".join([f"            {i+1}. {name} ({count}单)" for i, (name, count) in enumerate(top5_dest.items())])

            print("  - 已生成热门地点图表 (top_locations.png)")
        else:
            print("  - ⚠️ 警告：数据中缺少 dep_area 或 dest_area 列，跳过地点名称统计。")

        # ---------------------------------------------------------
        # [4/5] 空间分析 (地图)
        # ---------------------------------------------------------
        print("\n[4/5] 空间数据可视化...")

        temp_df = df.dropna(subset=['dep_latitude', 'dep_longitude']).copy()
        q_low = temp_df[['dep_latitude', 'dep_longitude']].quantile(0.001)
        q_high = temp_df[['dep_latitude', 'dep_longitude']].quantile(0.999)

        clean_df = temp_df[
            (temp_df['dep_latitude'] > q_low['dep_latitude']) &
            (temp_df['dep_latitude'] < q_high['dep_latitude']) &
            (temp_df['dep_longitude'] > q_low['dep_longitude']) &
            (temp_df['dep_longitude'] < q_high['dep_longitude'])
            ].copy()

        clean_df['distance_km'] = np.sqrt(
            (clean_df['dest_longitude'] - clean_df['dep_longitude'])**2 +
            (clean_df['dest_latitude'] - clean_df['dep_latitude'])**2
        ) * 111

        if len(clean_df) > 0:
            map_center = [clean_df['dep_latitude'].mean(), clean_df['dep_longitude'].mean()]
            dep_map = folium.Map(location=map_center, zoom_start=12, tiles='CartoDB positron')

            # 针对月度庞大数据，依旧随机抽样5万点生成热力地图，保证运行流畅不卡死
            if len(clean_df) > 50000:
                sample_data = clean_df.sample(50000)
            else:
                sample_data = clean_df

            heat_data = sample_data[['dep_latitude', 'dep_longitude']].values.tolist()
            HeatMap(heat_data, radius=8, blur=12, min_opacity=0.3).add_to(dep_map)
            dep_map.save('analysis_results2/maps/monthly_heatmap.html')

        # ---------------------------------------------------------
        # [5/5] 生成报告
        # ---------------------------------------------------------
        # 统计指标计算
        total_orders = len(df)
        days_span = df['date'].nunique()
        avg_daily = total_orders / days_span if days_span > 0 else 0

        hourly_counts = df.groupby('hour').size()
        peak_hour = hourly_counts.idxmax()
        peak_hour_vol = hourly_counts.max()

        # 时长统计
        avg_duration = df['trip_duration'].mean()
        max_duration = df['trip_duration'].max()
        min_duration = df['trip_duration'].min()

        weekend_days = df[df['weekday'] >= 5]['date'].nunique()
        weekday_days = df[df['weekday'] < 5]['date'].nunique()
        weekend_orders = df[df['weekday'] >= 5].shape[0]
        weekday_orders = df[df['weekday'] < 5].shape[0]

        avg_weekend = weekend_orders / weekend_days if weekend_days > 0 else 0
        avg_weekday = weekday_orders / weekday_days if weekday_days > 0 else 0

        report = f"""
        ================================================
        网约车月度运营分析报告
        ================================================
        分析时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        [1. 概览]
        ------------------------------------------------
        时间范围: {df['date'].min()} 至 {df['date'].max()} (共 {days_span} 天)
        总单量: {total_orders:,} 单
        日均单量: {avg_daily:.0f} 单/天
        
        [2. 热门地点 TOP 5]
        ------------------------------------------------
        【上车点 (出发)】
{top_dep_str}
        
        【下车点 (到达)】
{top_dest_str}
        
        [3. 峰值特征]
        ------------------------------------------------
        单日最高: {daily_stats['count'].max()} 单 ({daily_stats.loc[daily_stats['count'].idxmax(), 'day_label'].replace(chr(10), " ")})
        最热时段: {peak_hour}:00 - {peak_hour+1}:00 (本月累积共 {peak_hour_vol} 单)
        
        [4. 行程时长特征]
        ------------------------------------------------
        平均时长: {avg_duration:.1f} 分钟
        最长行程: {max_duration:.1f} 分钟
        最短行程: {min_duration:.1f} 分钟
        """

        print(report)
        with open('analysis_results2/report.txt', 'w', encoding='utf-8') as f:
            f.write(report)

        print("\n✅ 分析完成！请查看 analysis_results2 文件夹。")
        return True

    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    input_file = "data/finaldata2.csv"
    if os.path.exists(input_file):
        analyze_taxi_data_monthly(input_file)
    else:
        print(f"错误：找不到文件 {input_file}")