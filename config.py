# 文件名: download_assets.py
import os
import requests
import time

# 配置资源地址
# 注意：Tailwind 官方源通常最稳定，但如果下载慢，请看下文的"手动下载"方法
ASSETS = {
    "chart.js": "https://cdn.staticfile.net/Chart.js/3.9.1/chart.min.js",
    # 修改为官方源，这是独立运行版唯一的稳定地址
    "tailwindcss.js": "https://cdn.tailwindcss.com"
}

def download_file(url, filename):
    # 确保 static 文件夹存在
    current_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(current_dir, 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    filepath = os.path.join(static_dir, filename)

    # 如果文件已经存在且不为空，跳过（避免重复下载 Chart.js）
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        print(f"[跳过] 文件已存在: {filename}")
        return

    print(f"[下载中] 正在下载 {filename} ...")
    print(f"       源地址: {url}")

    try:
        # 添加 User-Agent 伪装成浏览器，防止被拦截
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        r = requests.get(url, headers=headers, timeout=30) # 增加超时时间到30秒
        r.raise_for_status()

        with open(filepath, 'wb') as f:
            f.write(r.content)
        print(f"[成功] 已保存到: {filepath}")
    except Exception as e:
        print(f"[失败] 无法下载 {filename}")
        print(f"       错误信息: {e}")
        print(f"       建议：请尝试【方法二：手动下载】")

if __name__ == "__main__":
    print("=== 开始更新静态资源 ===")
    for name, url in ASSETS.items():
        download_file(url, name)
    print("\n=== 检查完成 ===")

    # 检查 tailwind 是否真的下载成功了
    tw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'tailwindcss.js')
    if not os.path.exists(tw_path) or os.path.getsize(tw_path) < 1000:
        print("\n!!! 警告 !!!")
        print("tailwindcss.js 下载似乎失败了。")
        print("请务必执行下方的【方法二：手动下载】")
    else:
        print("所有文件准备就绪，可以运行 apply.py 了。")