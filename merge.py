import pandas as pd
import glob
import os

# 1. 配置路径
input_path = './data/'  # 如果文件在当前文件夹，请保持 './'；如果在 data 文件夹，请改为 './data/'
output_path = 'data'
output_file = os.path.join(output_path, 'jinan_taxi.csv')

# 2. 定义需要提取的 8 个列名
# 注意：请确保 CSV 文件中的列名与下面列表完全一致，如果文件中是“出发时间”等，请修改此处
target_columns = [
    'dep_time', 'dep_longitude', 'dep_latitude', 'dep_area',
    'dest_time', 'dest_longitude', 'dest_latitude', 'dest_area'
]

# 3. 如果输出目录不存在则创建
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 4. 获取所有待处理的 CSV 文件
# 匹配类似 "网约车交易订单信息.txt_0.csv" 的所有文件
all_files = glob.glob(os.path.join(input_path, "网约车交易订单信息*.csv"))

if not all_files:
    print("未找到匹配的文件，请检查文件路径和文件名。")
else:
    print(f"找到 {len(all_files)} 个文件，正在处理...")

    df_list = []
    # 修改后的读取部分
    for filename in all_files:
        try:
        # low_memory=False 消除警告
            df = pd.read_csv(filename, encoding='utf-8', low_memory=False)

        # 挑选 8 列
            df_filtered = df[target_columns].copy()

        # 【进阶建议】统一经纬度为数值类型，非数字的转为 NaN
            for col in ['dep_longitude', 'dep_latitude', 'dest_longitude', 'dest_latitude']:
                df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

            df_list.append(df_filtered)
            print(f"已处理: {filename}")


        except KeyError as e:
            print(f"错误：文件 {filename} 中缺少列: {e}")
        except Exception as e:
            print(f"读取文件 {filename} 时出错: {e}")

    # 5. 合并并保存
    if df_list:
        combined_df = pd.concat(df_list, axis=0, ignore_index=True)
        # 使用 utf_8_sig 确保 Excel 打开中文不乱码
        combined_df.to_csv(output_file, index=False, encoding='utf_8_sig')
        print("-" * 30)
        print(f"成功！合并后的文件包含 {len(combined_df)} 行数据。")
        print(f"结果已保存在: {output_file}")
        print(f"当前列名为: {list(combined_df.columns)}")