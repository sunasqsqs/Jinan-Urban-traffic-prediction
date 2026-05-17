import os
import sys
import subprocess
import warnings
import unicodedata

warnings.filterwarnings('ignore')

# 【终极防线 1】：强制单显卡运行，彻底杜绝多卡 (DataParallel) 梯度汇聚时的线程乱序误差！
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 【自动重启机制：确保 CUDA 环境与随机种子绝对纯净】
env_needs_update = False
env = os.environ.copy()

if env.get('PYTHONHASHSEED') != '42':
    env['PYTHONHASHSEED'] = '42'
    env_needs_update = True

if env.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    env_needs_update = True

if env_needs_update:
    subprocess.run([sys.executable] + sys.argv, env=env)
    sys.exit(0)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import json
import random
import time
import math

# ==========================================
# 解决 Matplotlib 中文显示问题 (明亮学术风格)
# ==========================================
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['text.color'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['grid.color'] = '#E2E8F0'

# ==========================================
# 0. 环境与随机种子锁定 (严格对照实验标准)
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 关闭自动寻优，强制 CUDA 卷积使用完全确定性的算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

# ==========================================
# 1. 实验配置参数
# ==========================================
class Config:
    data_path = 'data/finaldata.csv'  # 输入文件
    save_dir = 'real-time results/'             # 输出文件夹

    grid_size = (15, 15)
    time_interval = '30min'

    history_steps = 12
    future_steps = 1

    batch_size = 64
    epochs = 100

    train_ratio = 0.7
    val_ratio = 0.15

    input_dim = 3
    hidden_dim = 32
    num_layers = 3
    dropout = 0.15

    num_nodes = grid_size[0] * grid_size[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42

config = Config()
os.makedirs(config.save_dir, exist_ok=True)
set_seed(config.seed)

# ==========================================
# 视觉对齐辅助函数
# ==========================================
def get_display_width(text):
    width = 0
    for c in str(text):
        if unicodedata.east_asian_width(c) in ('F', 'W'):
            width += 2
        else:
            width += 1
    return width

def pad_str(text, target_width):
    text = str(text)
    display_width = get_display_width(text)
    padding = target_width - display_width
    return text + ' ' * (padding if padding > 0 else 0)

def format_table_row(items, widths):
    formatted = []
    for item, w in zip(items, widths):
        formatted.append(pad_str(item, w))
    return " | ".join(formatted)

# ==========================================
# 核心模块区 (RevIN, SwiGLU, RMSNorm)
# ==========================================
class RevIN_ST(nn.Module):
    def __init__(self, num_nodes, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, num_nodes, 1))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_nodes, 1))

    def forward(self, x, mode):
        if mode == 'norm':
            demand = x[..., 0:1]
            self.mean = demand.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(demand.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            demand_norm = (demand - self.mean) / self.stdev
            demand_norm = demand_norm * self.gamma + self.beta
            return torch.cat([demand_norm, x[..., 1:]], dim=-1)
        elif mode == 'denorm':
            x = (x - self.beta) / self.gamma
            x = x * self.stdev[:, -1:, :, :] + self.mean[:, -1:, :, :]
            return x

class SwiGLU(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim * 2
        self.w1 = nn.Linear(in_dim, hidden_dim)
        self.w2 = nn.Linear(in_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
        self.register_parameter('bias', nn.Parameter(torch.zeros(d)) if bias else None)

    def forward(self, x):
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.bias is not None:
            return normed * self.weight + self.bias
        return normed * self.weight

# ---------------------------------------------------------
# Mamba 状态空间模块
# ---------------------------------------------------------
try:
    from mamba_ssm import Mamba
    print(">> [系统] 成功载入原生 mamba-ssm 库！")

    class TS_MambaBlock(nn.Module):
        def __init__(self, d_model, expand=2, dropout=0.1):
            super().__init__()
            self.norm1 = RMSNorm(d_model)
            self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=expand)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            return x + self.drop(self.mamba(self.norm1(x)))

except ImportError:
    Mamba = None
    print(">> [系统] 未检测到 mamba_ssm，启用【Mock 单向 RNN】...")
    class TS_MambaBlock(nn.Module):
        def __init__(self, d_model, expand=2, dropout=0.1):
            super().__init__()
            self.norm1 = RMSNorm(d_model)
            self.seq_core = nn.GRU(d_model, d_model, bidirectional=False, batch_first=True)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            x_norm = self.norm1(x)
            seq_out, _ = self.seq_core(x_norm)
            return x + self.drop(seq_out)

def load_and_process_data():
    if not os.path.exists(config.data_path):
        raise FileNotFoundError(f"【致命错误】在路径 {config.data_path} 未找到数据集。请确保真实数据已就绪。")

    print(f"  [✓] 成功载入真实交通数据集: {config.data_path}")
    df = pd.read_csv(config.data_path)
    df['dep_time'] = pd.to_datetime(df['dep_time'])

    if 'grid_id' not in df.columns:
        lon_min, lon_max = 116.0, 118.0
        lat_min, lat_max = 36.0, 37.8
        lon_step = (lon_max - lon_min) / config.grid_size[0]
        lat_step = (lat_max - lat_min) / config.grid_size[1]
        def get_grid_id(row):
            x = int((row['dep_longitude'] - lon_min) / lon_step)
            y = int((row['dep_latitude'] - lat_min) / lat_step)
            x = max(0, min(x, config.grid_size[0] - 1))
            y = max(0, min(y, config.grid_size[1] - 1))
            return y * config.grid_size[0] + x
        df['grid_id'] = df.apply(get_grid_id, axis=1)

    lon_min, lon_max = 116.0, 118.0
    lat_min, lat_max = 36.0, 37.8
    lon_step = (lon_max - lon_min) / config.grid_size[0]
    lat_step = (lat_max - lat_min) / config.grid_size[1]

    df_agg = df.groupby([pd.Grouper(key='dep_time', freq=config.time_interval), 'grid_id']).size().unstack(fill_value=0)
    full_idx = pd.date_range(start=df_agg.index[0], end=df_agg.index[-1], freq=config.time_interval)
    df_agg = df_agg.reindex(full_idx, fill_value=0)

    for i in range(config.num_nodes):
        if i not in df_agg.columns:
            df_agg[i] = 0
    df_agg = df_agg[sorted(df_agg.columns)]

    hours = df_agg.index.hour + df_agg.index.minute / 60.0
    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)

    demand_data = df_agg.values.astype(np.float32)
    demand_log = np.log1p(demand_data)

    train_size = int(len(demand_log) * config.train_ratio)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(demand_log[:train_size])
    demand_norm = scaler.transform(demand_log)

    combined_data = np.zeros((demand_norm.shape[0], config.num_nodes, 3), dtype=np.float32)
    for t in range(demand_norm.shape[0]):
        combined_data[t, :, 0] = demand_norm[t]
        combined_data[t, :, 1] = hour_sin[t]
        combined_data[t, :, 2] = hour_cos[t]

    grid_meta = {
        'lon_min': lon_min, 'lon_step': lon_step,
        'lat_min': lat_min, 'lat_step': lat_step,
        'cols': config.grid_size[0], 'rows': config.grid_size[1]
    }
    return combined_data, scaler, grid_meta

def get_adjacency_matrix():
    adj = np.zeros((config.num_nodes, config.num_nodes), dtype=np.float32)
    rows, cols = config.grid_size
    for r in range(rows):
        for c in range(cols):
            curr = r * cols + c
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    adj[curr, nr * cols + nc] = 1.0

    adj = adj + np.eye(config.num_nodes)
    d = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_norm = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return torch.tensor(adj_norm, device=config.device, dtype=torch.float32)

def create_dataloaders(data):
    X, Y = [], []
    for i in range(len(data) - config.history_steps - config.future_steps + 1):
        X.append(data[i : i + config.history_steps])
        Y.append(data[i + config.history_steps : i + config.history_steps + config.future_steps, :, 0])

    X, Y = np.array(X), np.array(Y)

    train_end = int(len(X) * config.train_ratio)
    val_end = train_end + int(len(X) * config.val_ratio)

    X_train = torch.FloatTensor(X[:train_end])
    Y_train = torch.FloatTensor(Y[:train_end])
    X_val = torch.FloatTensor(X[train_end:val_end])
    Y_val = torch.FloatTensor(Y[train_end:val_end])
    X_test = torch.FloatTensor(X[val_end:])
    Y_test = torch.FloatTensor(Y[val_end:])

    def to_loader(x, y, shuffle=False):
        ds = TensorDataset(x, y)
        kwargs = {
            'batch_size': config.batch_size,
            'shuffle': shuffle,
            'num_workers': 0,
            'drop_last': False
        }
        return DataLoader(ds, **kwargs)

    return (to_loader(X_train, Y_train, True),
            to_loader(X_val, Y_val, False),
            to_loader(X_test, Y_test, False))

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, count_include_pad=False)

    def forward(self, x):
        x_t = x.permute(0, 2, 1)
        trend = self.moving_avg(x_t).permute(0, 2, 1)
        res = x - trend
        return res, trend

class AnchorReadout(nn.Module):
    def __init__(self, dim, seq_len):
        super().__init__()
        self.temporal_proj = nn.Linear(seq_len, 1)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        t_pool = self.temporal_proj(x.transpose(1, 2)).transpose(1, 2).squeeze(1)
        last_out = x[:, -1, :]
        fused = t_pool + last_out
        return self.norm(fused + self.proj(fused))

class SpatialDiffusion(nn.Module):
    def __init__(self, dim, num_nodes, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.weight0 = nn.Parameter(torch.eye(dim) + torch.randn(dim, dim) * 0.01)
        self.weight1 = nn.Parameter(torch.eye(dim) + torch.randn(dim, dim) * 0.01)
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        res = x
        x = self.norm(x)
        ax1 = torch.matmul(adj, x)
        out = torch.matmul(x, self.weight0) + torch.matmul(ax1, self.weight1)
        return res + self.alpha * self.drop(torch.nn.functional.gelu(out))

class ST_Mamba_Model(nn.Module):
    def __init__(self, adj=None):
        super().__init__()

        if adj is not None:
            self.register_buffer('adj_matrix', adj)

        self.h_dim = config.hidden_dim
        self.num_layers = config.num_layers

        self.revin = RevIN_ST(config.num_nodes)

        self.embedding = nn.Sequential(
            nn.Linear(config.input_dim, self.h_dim),
            nn.LayerNorm(self.h_dim),
            nn.GELU()
        )

        self.spatial_emb = nn.Parameter(torch.randn(1, 1, config.num_nodes, self.h_dim) * 0.02)
        self.temporal_emb = nn.Parameter(torch.randn(1, config.history_steps, 1, self.h_dim) * 0.02)
        self.st_pe = nn.Parameter(torch.randn(1, config.num_nodes, config.history_steps, self.h_dim) * 0.02)

        self.decomp = SeriesDecomp(kernel_size=3)
        self.trend_proj = nn.Linear(self.h_dim, self.h_dim)

        self.temporal_net = nn.ModuleList([
            TS_MambaBlock(d_model=self.h_dim, expand=2, dropout=config.dropout)
            for _ in range(self.num_layers)
        ])

        self.spatial_net = SpatialDiffusion(self.h_dim, config.num_nodes, dropout=config.dropout)
        self.s_transform = nn.Linear(self.h_dim, self.h_dim)
        self.st_norm = nn.LayerNorm(self.h_dim)

        self.fusion_gate = nn.Linear(self.h_dim * 2, self.h_dim)
        self.spatial_drop = nn.Dropout(config.dropout)

        self.cross_norm = nn.LayerNorm(self.h_dim)
        self.synergy_proj = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.h_dim // 2, self.h_dim)
        )

        self.temporal_readout = AnchorReadout(self.h_dim, config.history_steps)
        self.ar_full = nn.Linear(config.history_steps, 1)

        self.output_head = nn.Sequential(
            nn.LayerNorm(self.h_dim),
            SwiGLU(self.h_dim, self.h_dim, hidden_dim=self.h_dim * 2),
            nn.Dropout(config.dropout),
            nn.Linear(self.h_dim, 1)
        )

        self.conf_gate = nn.Linear(self.h_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'temporal_net' not in name and 'spatial_net' not in name:
                nn.init.xavier_uniform_(p)

        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)
        nn.init.xavier_uniform_(self.ar_full.weight)
        nn.init.zeros_(self.ar_full.bias)

        if hasattr(self, 'conf_gate'):
            nn.init.zeros_(self.conf_gate.weight)
            nn.init.constant_(self.conf_gate.bias, -1.0)

        if hasattr(self, 'fusion_gate'):
            nn.init.xavier_uniform_(self.fusion_gate.weight)
            nn.init.constant_(self.fusion_gate.bias, -3.0)

    def forward(self, x):
        B, T, N, C = x.shape

        x_norm = self.revin(x, 'norm')
        x_emb = self.embedding(x_norm) + self.spatial_emb + self.temporal_emb

        t_in = x_emb.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        res, trend = self.decomp(t_in)

        res_spatial = res.reshape(B, N, T, -1) + self.st_pe
        t_dyn_long = res_spatial.reshape(B, N * T, -1)

        for layer in self.temporal_net:
            t_dyn_long = layer(t_dyn_long)

        t_dyn = t_dyn_long.reshape(B, N, T, -1).reshape(B * N, T, -1)

        t_dyn_spatial = t_dyn.reshape(B, N, T, -1).permute(0, 2, 1, 3).reshape(B * T, N, -1)
        g_dyn_spatial = self.spatial_net(t_dyn_spatial, self.adj_matrix)
        g_dyn = g_dyn_spatial.reshape(B, T, N, -1).permute(0, 2, 1, 3).reshape(B * N, T, -1)

        g_feat = self.st_norm(self.s_transform(g_dyn))

        gate = torch.sigmoid(self.fusion_gate(torch.cat([t_dyn, g_feat], dim=-1)))

        cross_input = self.cross_norm(t_dyn * g_feat)
        st_synergy = self.synergy_proj(cross_input)
        spatial_info = self.spatial_drop(gate * (g_feat + st_synergy))
        t_dyn = t_dyn + spatial_info

        t_out = t_dyn + self.trend_proj(trend)
        t_feat = self.temporal_readout(t_out).reshape(B, N, -1)

        delta_norm = self.output_head(t_feat).reshape(B, 1, N, 1)
        conf = torch.sigmoid(self.conf_gate(t_feat)).reshape(B, 1, N, 1)

        history_demand = x_norm[..., 0]
        ar_base = self.ar_full(history_demand.transpose(1, 2)).transpose(1, 2).unsqueeze(-1)

        pred_norm = ar_base + delta_norm * conf
        pred_real = self.revin(pred_norm, 'denorm')

        return pred_real

PLOT_COLORS = {
    'Mamba-GNN': '#2563EB'
}

def plot_fusion_loss(history_dict, save_dir):
    if 'Mamba-GNN' not in history_dict: return
    hist = history_dict['Mamba-GNN']
    plt.figure(figsize=(10, 6))
    plt.plot(hist['train_loss'], label='训练损失 (Training Loss)', color='#2563EB', linewidth=2, alpha=0.9)
    plt.plot(hist['val_loss'], label='验证损失 (Validation Loss)', color='#16A34A', linewidth=2, linestyle='--', alpha=0.9)
    plt.title("Mamba-GNN 模型训练损失", fontsize=16)
    plt.xlabel("轮数 (Epochs)", fontsize=12)
    plt.ylabel("损失 (Loss)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5, color='#E2E8F0')
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(save_dir, 'fusion_model_loss.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_total_demand(all_preds, all_trues, save_dir, prefix=""):
    if not all_preds: return
    plt.figure(figsize=(15, 6))
    first_key = list(all_trues.keys())[0]
    total_true = np.sum(all_trues[first_key], axis=1)
    plot_len = min(len(total_true), 200)

    plt.plot(total_true[:plot_len], label='真实值 (Ground Truth)', color='#475569', linewidth=3, alpha=0.6, linestyle='--')

    for name in all_preds.keys():
        preds = all_preds[name]
        total_pred = np.sum(preds, axis=1)
        plt.plot(total_pred[:plot_len], label=name, color=PLOT_COLORS.get(name, '#2563EB'),
                 linewidth=3.5, linestyle='-', alpha=0.95)

    title_str = f"[{prefix}] 总需求量预测对比" if prefix else "总需求量预测对比"
    plt.title(title_str, fontsize=16)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5, color='#E2E8F0')
    plt.tight_layout()
    filename = f"{prefix}_total_demand_comparison.png" if prefix else "total_demand_comparison.png"
    plt.savefig(os.path.join(save_dir, filename.replace(" ", "_")), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_scatter_fit(all_preds, all_trues, save_dir, prefix=""):
    if not all_preds: return
    plt.figure(figsize=(6, 6))

    for i, (name, preds) in enumerate(all_preds.items()):
        trues = all_trues[name].flatten()
        preds = preds.flatten()
        idx = np.random.choice(len(trues), min(10000, len(trues)), replace=False)
        plt.scatter(trues[idx], preds[idx], alpha=0.4, color=PLOT_COLORS.get(name, '#2563EB'), s=5)
        max_val = max(trues[idx].max(), preds[idx].max())
        plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='理想拟合线', color='#EF4444')

        plt.title(f"{name} 拟合分析", fontsize=14, color='#333333')
        plt.xlabel("真实值", fontsize=12)
        plt.ylabel("预测值", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5, color='#E2E8F0')
        plt.legend()

    plt.tight_layout()
    filename = f"{prefix}_goodness_of_fit.png" if prefix else "goodness_of_fit.png"
    plt.savefig(os.path.join(save_dir, filename.replace(" ", "_")), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_spatial_error(preds, trues, grid_meta, name, save_dir):
    node_mae = np.mean(np.abs(trues - preds), axis=0)
    error_matrix = node_mae.reshape((grid_meta['rows'], grid_meta['cols']))
    plt.figure(figsize=(8, 6))
    plt.imshow(error_matrix, cmap='YlOrRd', origin='lower', aspect='auto')
    plt.colorbar(label='平均绝对误差 (MAE)')
    plt.title(f"{name} 空间误差分布", fontsize=14)
    plt.xlabel("经度网格索引")
    plt.ylabel("纬度网格索引")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'spatial_error_map_{name}.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_error_distribution(all_preds, all_trues, save_dir, prefix=""):
    if not all_preds: return
    plt.figure(figsize=(12, 6))

    for name, preds in all_preds.items():
        errors = (all_trues[name] - preds).flatten()
        # 过滤极端异常值以保证图像比例美观
        errors = errors[(errors >= -15) & (errors <= 15)]
        plt.hist(errors, bins=100, alpha=0.5, label=name, color=PLOT_COLORS.get(name, '#2563EB'), density=True, histtype='stepfilled', edgecolor='white', linewidth=2)

    plt.axvline(x=0, color='#334155', linestyle='--', linewidth=2)
    title_str = f"[{prefix}] 误差分布" if prefix else "误差分布"
    plt.title(title_str, fontsize=16)
    plt.xlabel("误差值 (真实值 - 预测值)", fontsize=12)
    plt.ylabel("密度 (Density)", fontsize=12)
    plt.legend(fontsize=10, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5, color='#E2E8F0')
    plt.tight_layout()
    filename = f"{prefix}_error_distribution.png" if prefix else "error_distribution.png"
    plt.savefig(os.path.join(save_dir, filename.replace(" ", "_")), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_epoch_metrics(history_dict, save_dir):
    for name, hist in history_dict.items():
        if 'train_mse' not in hist or not hist['train_mse']: continue
        epochs = range(1, len(hist['train_mse']) + 1)
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"{name} - 训练指标变化曲线", fontsize=18, weight='bold', color='#333333')

        axs[0, 0].plot(epochs, hist['train_acc'], label='训练 Acc', color='#2563EB', linewidth=2)
        axs[0, 0].plot(epochs, hist['val_acc'], label='验证 Acc', color='#16A34A', linewidth=2, linestyle='--')
        axs[0, 0].plot(epochs, hist['test_acc'], label='测试 Acc', color='#D97706', linewidth=2, linestyle='-.')
        axs[0, 0].set_title('准确率 (Accuracy)', fontsize=14)
        axs[0, 0].legend()
        axs[0, 0].grid(True, linestyle='--', alpha=0.5, color='#E2E8F0')

        axs[0, 1].plot(epochs, hist['train_mae'], label='训练 MAE', color='#2563EB', linewidth=2)
        axs[0, 1].plot(epochs, hist['val_mae'], label='验证 MAE', color='#16A34A', linewidth=2, linestyle='--')
        axs[0, 1].plot(epochs, hist['test_mae'], label='测试 MAE', color='#D97706', linewidth=2, linestyle='-.')
        axs[0, 1].set_title('平均绝对误差 (MAE)', fontsize=14)
        axs[0, 1].legend()
        axs[0, 1].grid(True, linestyle='--', alpha=0.5, color='#E2E8F0')

        axs[1, 0].plot(epochs, hist['train_mse'], label='训练 MSE', color='#2563EB', linewidth=2)
        axs[1, 0].plot(epochs, hist['val_mse'], label='验证 MSE', color='#16A34A', linewidth=2, linestyle='--')
        axs[1, 0].plot(epochs, hist['test_mse'], label='测试 MSE', color='#D97706', linewidth=2, linestyle='-.')
        axs[1, 0].set_title('均方误差 (MSE)', fontsize=14)
        axs[1, 0].legend()
        axs[1, 0].grid(True, linestyle='--', alpha=0.5, color='#E2E8F0')

        train_r2 = [max(x, -1.0) for x in hist['train_r2']]
        val_r2 = [max(x, -1.0) for x in hist['val_r2']]
        test_r2 = [max(x, -1.0) for x in hist['test_r2']]
        axs[1, 1].plot(epochs, train_r2, label='训练 R2', color='#2563EB', linewidth=2)
        axs[1, 1].plot(epochs, val_r2, label='验证 R2', color='#16A34A', linewidth=2, linestyle='--')
        axs[1, 1].plot(epochs, test_r2, label='测试 R2', color='#D97706', linewidth=2, linestyle='-.')
        axs[1, 1].set_title('决定系数 (R2 Score)', fontsize=14)
        axs[1, 1].legend()
        axs[1, 1].grid(True, linestyle='--', alpha=0.5, color='#E2E8F0')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f'{name}_all_metrics_curves.png'), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

def compute_mape(y_true, y_pred, threshold=5.0):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mask = y_true > threshold
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-6) * 100


# ==========================================
# 基于模型真实预测结果提取并分析 TOP-K 热点区域 (动态生成具体信息)
# ==========================================
def analyze_top_predicted_regions(all_preds, all_trues, grid_meta, save_dir, prefix="", top_k=5):
    if 'Mamba-GNN' not in all_preds: return []

    preds = all_preds['Mamba-GNN']
    trues = all_trues['Mamba-GNN']

    # 基于预测数据计算所有网格的总流量
    total_vol = np.sum(preds, axis=0)
    sorted_indices = np.argsort(-total_vol)
    top_indices = sorted_indices[:top_k]

    analysis_report = []
    plot_len = min(len(preds), 200)

    for rank, grid_id in enumerate(top_indices):
        grid_id = int(grid_id)

        # 换算网格中心经纬度坐标
        cols = grid_meta['cols']
        y = grid_id // cols
        x = grid_id % cols
        lon_center = grid_meta['lon_min'] + (x + 0.5) * grid_meta['lon_step']
        lat_center = grid_meta['lat_min'] + (y + 0.5) * grid_meta['lat_step']
        region_name = f"Grid_{grid_id}"

        # 提取单独网格的数据
        true_series = trues[:, grid_id]
        pred_series = preds[:, grid_id]

        # 单独计算该区域指标
        mae = mean_absolute_error(true_series, pred_series)
        rmse = np.sqrt(mean_squared_error(true_series, pred_series))
        t_vol = float(np.sum(true_series))
        p_vol = float(np.sum(pred_series))

        # --- 1. 为该热点网格单独绘制【需求预测对比图】 ---
        plt.figure(figsize=(10, 4))
        plt.plot(true_series[:plot_len], label='真实流量 (Ground Truth)', color='#475569', linewidth=2, alpha=0.7, linestyle='--')
        plt.plot(pred_series[:plot_len], label='Mamba-GNN 预测', color='#2563EB', linewidth=2.5, alpha=0.9)
        plt.title(f"TOP {rank+1} 热点区域 [网格 {grid_id}] 需求走势 (经度:{lon_center:.3f}, 纬度:{lat_center:.3f})", fontsize=14, color='#333333')
        plt.xlabel("时间步 (Time Steps)", fontsize=12)
        plt.ylabel("订单需求量", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5, color='#E2E8F0')
        plt.legend(loc='upper right', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_top{rank+1}_grid{grid_id}_demand.png'), dpi=300, facecolor='white')
        plt.close()

        # --- 2. 为该热点网格单独绘制【误差分布直方图】 ---
        errors = true_series - pred_series
        # 过滤极端异常值以保证图像比例美观
        errors = errors[(errors >= -15) & (errors <= 15)]
        plt.figure(figsize=(8, 4))
        plt.hist(errors, bins=40, alpha=0.7, color='#D97706', density=True, histtype='stepfilled', edgecolor='white')
        plt.axvline(x=0, color='#334155', linestyle='--', linewidth=2)
        plt.title(f"TOP {rank+1} 热点区域 [网格 {grid_id}] 预测误差分布", fontsize=14, color='#333333')
        plt.xlabel("误差值 (真实值 - 预测值)", fontsize=12)
        plt.ylabel("密度 (Density)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5, color='#E2E8F0')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_top{rank+1}_grid{grid_id}_error.png'), dpi=300, facecolor='white')
        plt.close()

        # --- 3. 将详细指标和时间步序列保存到报告数组中 ---
        analysis_report.append({
            "rank": rank + 1,
            "grid_id": grid_id,
            "lon_center": round(lon_center, 4),
            "lat_center": round(lat_center, 4),
            "metrics": {
                "MAE": round(float(mae), 4),
                "RMSE": round(float(rmse), 4),
                "total_true_volume": int(t_vol),
                "total_predicted_volume": int(p_vol)
            },
            "time_series_data": {
                "true_values": [round(float(v), 2) for v in true_series[:plot_len]],
                "predicted_values": [round(float(v), 2) for v in pred_series[:plot_len]]
            }
        })

    return analysis_report


# ==========================================
# 5. 实验主逻辑
# ==========================================
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_model(model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_model(model, path)

    def save_model(self, model, path):
        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state, path)

class RobustTrafficLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=1.0)
        self.mae = nn.L1Loss()
        self.alpha = alpha

    def forward(self, pred, true):
        return self.huber(pred, true) + self.alpha * self.mae(pred, true)

def run_exp(name, model, loaders, scaler):
    set_seed(config.seed)

    train_loader, val_loader, test_loader = loaders

    weight_decay = 1e-4
    lr = 1.0e-3
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-5)

    criterion = RobustTrafficLoss(alpha=0.5)
    early_stopping = EarlyStopping(patience=5, delta=1e-6)

    scaler_amp = torch.cuda.amp.GradScaler(enabled=False)

    save_path = os.path.join(config.save_dir, f'best_model_{name}.pt')

    history = {
        'train_loss': [], 'val_loss': [],
        'train_mse': [], 'val_mse': [], 'test_mse': [],
        'train_mae': [], 'val_mae': [], 'test_mae': [],
        'train_r2': [], 'val_r2': [], 'test_r2': [],
        'train_acc': [], 'val_acc': [], 'test_acc': []
    }

    def calculate_metrics_from_tensors(p_list, t_list):
        pred_norm = np.concatenate(p_list)
        true_norm = np.concatenate(t_list)

        pred_norm = np.clip(pred_norm, 0.0, 1.0)
        true_norm = np.clip(true_norm, 0.0, 1.0)

        pred_log = scaler.inverse_transform(pred_norm)
        true_log = scaler.inverse_transform(true_norm)

        pred_log = np.clip(pred_log, a_min=-10.0, a_max=20.0)

        p_final = np.maximum(np.expm1(pred_log), 0)
        t_final = np.maximum(np.expm1(true_log), 0)

        mse = mean_squared_error(t_final, p_final)
        mae = mean_absolute_error(t_final, p_final)
        r2 = r2_score(t_final.flatten(), p_final.flatten())
        wmape = compute_wmape(t_final, p_final)
        acc = max(0.0, 100.0 - wmape)
        return mse, mae, r2, acc

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  > [{name}] 初始化完成 | 模型参数量: {total_params:,}")

    total_train_time = 0.0
    epochs_run = 0

    for epoch in range(config.epochs):
        epoch_start_time = time.time()

        model.train()
        t_loss = 0
        train_p, train_t = [], []
        for bx, by in train_loader:
            bx, by = bx.to(config.device), by.to(config.device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=False):
                p = model(bx)
                loss = criterion(p.squeeze(), by.squeeze())

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            t_loss += loss.item()
            train_p.append(p.detach().cpu().reshape(-1, config.num_nodes))
            train_t.append(by.cpu().reshape(-1, config.num_nodes))

        model.eval()
        v_loss = 0
        val_p, val_t = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(config.device), by.to(config.device)

                with torch.cuda.amp.autocast(enabled=False):
                    p = model(bx)
                    v_loss += criterion(p.squeeze(), by.squeeze()).item()
                val_p.append(p.cpu().reshape(-1, config.num_nodes))
                val_t.append(by.cpu().reshape(-1, config.num_nodes))

        test_p, test_t = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(config.device), by.to(config.device)

                with torch.cuda.amp.autocast(enabled=False):
                    p = model(bx)
                test_p.append(p.cpu().reshape(-1, config.num_nodes))
                test_t.append(by.cpu().reshape(-1, config.num_nodes))

        tr_mse, tr_mae, tr_r2, tr_acc = calculate_metrics_from_tensors(train_p, train_t)
        v_mse, v_mae, v_r2, v_acc = calculate_metrics_from_tensors(val_p, val_t)
        te_mse, te_mae, te_r2, te_acc = calculate_metrics_from_tensors(test_p, test_t)

        avg_t = t_loss / len(train_loader)
        avg_v = v_loss / len(val_loader)

        history['train_loss'].append(avg_t)
        history['val_loss'].append(avg_v)

        history['train_mse'].append(tr_mse); history['val_mse'].append(v_mse); history['test_mse'].append(te_mse)
        history['train_mae'].append(tr_mae); history['val_mae'].append(v_mae); history['test_mae'].append(te_mae)
        history['train_r2'].append(tr_r2);   history['val_r2'].append(v_r2);   history['test_r2'].append(te_r2)
        history['train_acc'].append(tr_acc); history['val_acc'].append(v_acc); history['test_acc'].append(te_acc)

        scheduler.step()
        early_stopping(avg_v, model, save_path)

        epoch_duration = time.time() - epoch_start_time
        total_train_time += epoch_duration
        epochs_run += 1

        if early_stopping.early_stop:
            print(f"  -> [{name}] 在第 {epoch+1} 轮触发提前停止 (Early Stopping)")
            break

        if (epoch+1) % 5 == 0 or (epoch+1) == config.epochs:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  -> Epoch [{epoch+1:03d}/{config.epochs:03d}] | Val Loss: {avg_v:.5f} | Val MAE: {v_mae:.4f} | LR: {current_lr:.2e} | 耗时: {epoch_duration:.2f}s")

    state_dict = torch.load(save_path)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    model.eval()

    all_p, all_t = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(config.device), by.to(config.device)
            with torch.cuda.amp.autocast(enabled=False):
                p = model(bx)
            all_p.append(p.cpu().reshape(-1, config.num_nodes))
            all_t.append(by.cpu().reshape(-1, config.num_nodes))

    pred_norm = np.concatenate(all_p)
    true_norm = np.concatenate(all_t)

    pred_norm = np.clip(pred_norm, 0.0, 1.0)
    true_norm = np.clip(true_norm, 0.0, 1.0)

    pred_log = scaler.inverse_transform(pred_norm)
    true_log = scaler.inverse_transform(true_norm)

    p_final = np.maximum(np.expm1(pred_log), 0)
    t_final = np.maximum(np.expm1(true_log), 0)

    avg_epoch_time = total_train_time / epochs_run

    return history, p_final, t_final, -early_stopping.best_score, total_params, avg_epoch_time

TIME_COMPLEXITY_MAP = {
    'Mamba-GNN': 'O(T·N² + B·(T·N)·d) [并行扫描]',
}

def main():
    print("="*100)
    print(" 🚀 纯净版 Mamba-GNN 模型环境初始化完成！")
    print("="*100)

    data, scaler, grid_meta = load_and_process_data()
    loaders = create_dataloaders(data)
    adj = get_adjacency_matrix()

    name = 'Mamba-GNN'
    num_runs = 1

    results = {}
    all_hist, all_pred, all_true = {} , {}, {}

    final_report = {
        "evaluation_groups": {},
        "loss_history": {},
        "predictions_time_series": {},
        "top_predicted_regions_analysis": []  # 用来存放模型预测出的Top 5详细序列和各项误差值
    }

    print("="*80)
    print(f"🔹 [实验进行中] 模型名称: {name} (运行 1 次)")
    print("-" * 80)

    run_metrics = {'mse': [], 'rmse': [], 'mae': [], 'mape': [], 'wmape': [], 'r2': [], 'time': []}
    best_mae = float('inf')
    best_hist, best_p, best_t = None, None, None
    final_params = 0

    for i in range(num_runs):
        print(f"\n   >>> 正在执行第 {i+1}/{num_runs} 次独立运行...")
        set_seed(config.seed)

        model = ST_Mamba_Model(adj=adj).to(config.device)
        h, p, t, val_loss, params, avg_time = run_exp(name, model, loaders, scaler)

        mse = mean_squared_error(t, p)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(t, p)
        r2 = r2_score(t.flatten(), p.flatten())
        mape = compute_mape(t, p, threshold=5.0)
        wmape = compute_wmape(t, p)

        run_metrics['mse'].append(mse)
        run_metrics['rmse'].append(rmse)
        run_metrics['mae'].append(mae)
        run_metrics['mape'].append(mape)
        run_metrics['wmape'].append(wmape)
        run_metrics['r2'].append(r2)
        run_metrics['time'].append(avg_time)
        final_params = params

        if mae < best_mae:
            best_mae = mae
            best_hist, best_p, best_t = h, p, t

    results[name] = {
        "mse": f"{run_metrics['mse'][0]:.2f}",
        "rmse": f"{run_metrics['rmse'][0]:.2f}",
        "mae": f"{run_metrics['mae'][0]:.2f}",
        "mape": f"{run_metrics['mape'][0]:.2f}",
        "wmape": f"{run_metrics['wmape'][0]:.2f}",
        "r2": f"{run_metrics['r2'][0]:.4f}",
        "parameters": int(final_params),
        "avg_epoch_time_s": f"{run_metrics['time'][0]:.2f}"
    }

    all_hist[name] = best_hist
    all_pred[name] = best_p
    all_true[name] = best_t

    final_report["loss_history"][name] = {
        "train": [float(x) for x in best_hist['train_loss']],
        "val": [float(x) for x in best_hist['val_loss']]
    }

    widths = [16, 14, 14, 14, 14, 14, 15, 10, 12, 45]
    headers = ['Model', 'MSE', 'RMSE', 'MAE', 'MAPE', 'WMAPE', 'R2', 'Params', 'Time/Ep(s)', 'Complexity']
    line_length = sum(widths) + len(widths) * 3 - 1

    print("\n\n" + "="*line_length)
    print(" 严谨学术模型评估与对比报告")
    print("="*line_length)

    print(f"\n>> 专属模型性能报告 (Mamba-GNN)")
    print("-" * line_length)
    print(format_table_row(headers, widths))
    print("-" * line_length)

    m = results[name]
    cplx = TIME_COMPLEXITY_MAP.get(name, 'N/A')
    columns = [
        name,
        str(m['mse']), str(m['rmse']), str(m['mae']),
        str(m['mape']), str(m['wmape']), str(m['r2']),
        str(m['parameters']), str(m['avg_epoch_time_s']),
        cplx
    ]
    print(format_table_row(columns, widths))
    print("-" * line_length)

    final_report["evaluation_groups"]["Mamba_GNN_Single"] = {name: m}

    # 绘制全网级别的明亮风格图表
    plot_total_demand(all_pred, all_true, config.save_dir, prefix="Mamba-GNN_Single")
    plot_scatter_fit(all_pred, all_true, config.save_dir, prefix="Mamba-GNN_Single")
    plot_error_distribution(all_pred, all_true, config.save_dir, prefix="Mamba-GNN_Single")
    plot_fusion_loss(all_hist, config.save_dir)
    plot_spatial_error(all_pred['Mamba-GNN'], all_true['Mamba-GNN'], grid_meta, 'Mamba-GNN', config.save_dir)
    plot_epoch_metrics(all_hist, config.save_dir)

    # 记录全网需求汇总序列
    gt_series = np.sum(all_true['Mamba-GNN'], axis=1).tolist()
    final_report["predictions_time_series"]["ground_truth"] = [float(x) for x in gt_series]
    pred_series = np.sum(all_pred['Mamba-GNN'], axis=1).tolist()
    final_report["predictions_time_series"]["prediction_mamba_gnn"] = [float(x) for x in pred_series]


    # ==============================================================
    # 重点：动态寻找模型预测出的 TOP 5 区域，分别制图并保存具体数值入 JSON
    # ==============================================================
    print("\n" + "="*line_length)
    print(" 📍 正在提取由模型真实预测出的流量最高 TOP 5 热点区域...")
    print("-" * line_length)

    top_predicted_regions_data = analyze_top_predicted_regions(
        all_pred, all_true, grid_meta, config.save_dir, prefix="Mamba-GNN_Single", top_k=5
    )
    final_report["top_predicted_regions_analysis"] = top_predicted_regions_data

    headers_hot = ['Rank', 'Grid ID', '预测总流量', '真实总流量', 'MAE', 'RMSE']
    widths_hot = [6, 8, 14, 14, 10, 10]
    line_hot_length = sum(widths_hot) + len(widths_hot) * 3 - 1

    print(format_table_row(headers_hot, widths_hot))
    print("-" * line_hot_length)

    for item in top_predicted_regions_data:
        row_hot = [
            str(item['rank']),
            str(item['grid_id']),
            str(item['metrics']['total_predicted_volume']),
            str(item['metrics']['total_true_volume']),
            f"{item['metrics']['MAE']:.2f}",
            f"{item['metrics']['RMSE']:.2f}"
        ]
        print(format_table_row(row_hot, widths_hot))
    print("="*line_hot_length)
    print(f"👉 TOP 5 区域的 [预测走势图] 及 [误差分布直方图] 已分别为这 5 个区域独立生成。")


    # 保存最终详细 JSON 报告
    json_path = os.path.join(config.save_dir, 'experiment_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    print(f"\n[可视化] 实验评估图表已生成至目录: {os.path.abspath(config.save_dir)}")
    print(f"[文件] 结构化报告 JSON 已生成: {json_path} (含 TOP-5 具体时间序列和独立误差项)")
    print(f"[完成] 所有深度学习评估流程安全结束！")

if __name__ == '__main__':
    main()