import pandas as pd

# ==========================================
# 1. 配置路径
# ==========================================
input_file = 'data/jinan_taxi.csv'      # 原始输入文件
output_file = 'data/finaldata.csv'      # 最终输出文件

# 筛选日期范围
start_date = '2024-07-01'
end_date_cutoff = '2024-07-07 23:59:59'

# 空间过滤范围
lon_range = (116, 118)
lat_range = (36, 37.8)

print("=" * 30)
print("开始执行处理流程...")
print(f"读取原始文件: {input_file}")
df = pd.read_csv(input_file, low_memory=False, dtype=str)
raw_count = len(df)
print(f">>> 原始总行数: {raw_count}")

# 3.1 经纬度处理：转换为浮点数并除以 1,000,000
print("正在处理经纬度...")
geo_cols = ['dep_longitude', 'dep_latitude', 'dest_longitude', 'dest_latitude']
for col in geo_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce') / 1_000_000

# 3.2 空间过滤（济南范围）
print("正在执行空间过滤...")
df = df[
    (df['dep_longitude'].between(*lon_range)) &
    (df['dep_latitude'].between(*lat_range)) &
    (df['dest_longitude'].between(*lon_range)) &
    (df['dest_latitude'].between(*lat_range))
    ].dropna(subset=geo_cols)
print(f">>> 空间过滤后剩余: {len(df)} 行")

# 3.3 时间处理：修正 20240531234536.0 这种格式
print("正在处理时间格式...")
def parse_custom_time(series):
    # 先转为字符串，去掉末尾的 '.0'，然后按格式转换
    s = series.str.split('.').str[0]
    return pd.to_datetime(s, format='%Y%m%d%H%M%S', errors='coerce')

df['dep_time'] = parse_custom_time(df['dep_time'])
df['dest_time'] = parse_custom_time(df['dest_time'])

# 剔除无法解析的时间
df = df.dropna(subset=['dep_time', 'dest_time'])
print(f">>> 时间格式修正后剩余: {len(df)} 行")

# 3.4 地点名称处理：去除末尾多余的空格
print("正在清理文本空格...")
df['dep_area'] = df['dep_area'].str.strip()
df['dest_area'] = df['dest_area'].str.strip()

# 4.1 日期筛选
print(f"正在筛选日期范围: {start_date} 至 {end_date_cutoff} ...")
mask = (df['dep_time'] >= start_date) & (df['dep_time'] <= end_date_cutoff)
df = df.loc[mask].copy()

# 4.2 去除重复数据
print("正在去除重复数据...")
before_dedup = len(df)
df.drop_duplicates(inplace=True)
after_dedup = len(df)
print(f"去重完成：删除 {before_dedup - after_dedup} 条重复数据。")

# 4.3 排序（按上车时间先后排序）
print("正在按时间排序...")
df = df.sort_values(by='dep_time')

print("-" * 30)
if len(df) > 0:
    print(f"处理成功！")
    print(f"最终保留有效行数: {len(df)}")
    print(f"最终数据清洗留存率: {(len(df) / raw_count * 100):.2f}%")
    print(f"时间范围: {df['dep_time'].min()} 至 {df['dep_time'].max()}")

    df.to_csv(output_file, index=False, encoding='utf_8_sig')
    print(f"结果已保存至: {output_file}")
else:
    print("警告：处理后未找到任何数据，请检查原始数据或筛选条件。")