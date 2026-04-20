import sys
import torch
import subprocess
import importlib.metadata

def check_environment():
    print("\n" + "=" * 50)
    print(">> 深度学习环境配置检测报告")
    print("=" * 50)

    # 1. Python 版本
    print(f"{'Python 版本':<20}: {sys.version.split()[0]}")

    # 2. PyTorch 版本
    print(f"{'PyTorch 版本':<20}: {torch.__version__}")

    # 3. CUDA 版本 (PyTorch 编译依赖)
    # 这是 PyTorch 实际使用的 CUDA 版本，最关键
    print(f"{'CUDA 版本 ':<20}: {torch.version.cuda}")

    # 4. mamba_ssm 版本 (深度学习库)
    try:
        mamba_lib_version = importlib.metadata.version("mamba_ssm")
        print(f"{'mamba_ssm 库版本':<20}: {mamba_lib_version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{'mamba_ssm 库版本':<20}: 未安装 (PackageNotFoundError)")

    print("-" * 50)

    # 6. GPU 状态检测
    gpu_available = torch.cuda.is_available()
    print(f"{'GPU 是否可用':<20}: {'✅ 是' if gpu_available else '❌ 否'}")

    if gpu_available:
        try:
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)

            print(f"{'GPU 数量':<20}: {gpu_count}")
            print(f"{'当前 GPU 名称':<20}: {gpu_name}")

            # 简单的显存测试
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f"{'当前显存状态':<20}: 空闲 {free_mem/1024**3:.2f} GB / 总计 {total_mem/1024**3:.2f} GB")
        except Exception as e:
            print(f"{'GPU 信息获取失败':<20}: {e}")
    else:
        print(f"{'提示':<20}: 当前 PyTorch 运行在 CPU 模式，无法使用显卡加速。")

    print("=" * 50 + "\n")

if __name__ == "__main__":
    check_environment()