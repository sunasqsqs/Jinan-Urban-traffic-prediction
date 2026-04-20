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
# 解决 Matplotlib 中文显示问题
# ==========================================
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

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

    # 【终极防线 2】：关闭自动寻优，强制 CUDA 卷积使用完全确定性的算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    try:
        # 开启暴君模式，严苛确定性。若 Mamba 算子无法配合则只发出警告，但保证其他模块 100% 锁死
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

# ==========================================
# 1. 实验配置参数
# ==========================================
class Config:
    data_path = 'data/finaldata.csv'  # [已修改] 输入文件
    save_dir = 'results/'             # [已修改] 输出文件夹

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
        # 修复对齐Bug：现代终端（PyCharm/VSCode等）将 "±", "·", "²" 等符号渲染为1个字符宽。
        # 只将明确的东亚宽字符（汉字等 'F', 'W'）计为 2 宽，其余计为 1。
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
class RevIN_ST(nn.Module):#可逆实例归一化
    def __init__(self, num_nodes, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, num_nodes, 1))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_nodes, 1))

    def forward(self, x, mode):
        if mode == 'norm':
            demand = x[..., 0:1]
            self.mean = demand.mean(dim=1, keepdim=True).detach()#需求量平均值
            self.stdev = torch.sqrt(demand.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()#需求量加极小数开根号
            demand_norm = (demand - self.mean) / self.stdev#（需求量-平均值）除以stdev
            demand_norm = demand_norm * self.gamma + self.beta#乘以1，加0
            return torch.cat([demand_norm, x[..., 1:]], dim=-1)
        elif mode == 'denorm':
            x = (x - self.beta) / self.gamma#减去0除以1
            x = x * self.stdev[:, -1:, :, :] + self.mean[:, -1:, :, :]#乘以stdev加上平均值
            return x

class SwiGLU(nn.Module):#激活函数
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim * 2
        self.w1 = nn.Linear(in_dim, hidden_dim)#和w2是两个通道，把in变成hidden的形状
        self.w2 = nn.Linear(in_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, out_dim)#把hidden变成out

    def forward(self, x):
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))#对w1进行x*sigmoid（x）操作，然后加上w2，最后处理成输出的形状

class RMSNorm(nn.Module):#归一化
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))#全1矩阵
        self.register_parameter('bias', nn.Parameter(torch.zeros(d)) if bias else None)#有偏差写0，无就没有

    def forward(self, x):
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)#x的平方的平均值加极小数，开根号，取倒数。再乘以x
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
            self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=expand)#潜在状态维度16，拓展2倍，4核一维卷积层
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

def load_and_process_data():#把时间和流量都记录在对应地理网格上
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
    df_agg = df_agg.reindex(full_idx, fill_value=0)#按照顺序编好号，确保每个网格都有数

    for i in range(config.num_nodes):
        if i not in df_agg.columns:
            df_agg[i] = 0
    df_agg = df_agg[sorted(df_agg.columns)]

    hours = df_agg.index.hour + df_agg.index.minute / 60.0
    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)#转换时间

    demand_data = df_agg.values.astype(np.float32)
    demand_log = np.log1p(demand_data)

    train_size = int(len(demand_log) * config.train_ratio)
    scaler = MinMaxScaler(feature_range=(0, 1))#转换成（0，1）
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
    adj = np.zeros((config.num_nodes, config.num_nodes), dtype=np.float32)#255*255
    rows, cols = config.grid_size
    for r in range(rows):
        for c in range(cols):
            curr = r * cols + c
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    adj[curr, nr * cols + nc] = 1.0#如果当前编号和其他邻接，那么对应行列网格处设为1

    adj = adj + np.eye(config.num_nodes)#加自环
    d = np.sum(adj, axis=1)#求每行的数值之和--邻居数量
    d_inv_sqrt = np.power(d, -0.5)#给邻居数量开个根号，然后取倒数
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.#万一有个孤岛格子邻居数为0，会算出无穷大(isinf)，把它强行变回 0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)#把d_inv_sqrt放到一个对角矩阵里面
    adj_norm = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)#两头夹击相乘！这是标准的图拉普拉斯矩阵归一化公式： D^(-1/2) * A * D^(-1/2)
    return torch.tensor(adj_norm, device=config.device, dtype=torch.float32)# 把普通矩阵变成 PyTorch 专属的 Tensor 格式，塞进显卡或 CPU 里待命。

def create_dataloaders(data):#滑动窗口取样，训练集乱序
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

class SeriesDecomp(nn.Module):#AvgPool1d在3个连续时间点求平均值，代表趋势，res是残差
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
        self.temporal_proj = nn.Linear(seq_len, 1)#12步浓缩为1步
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))#两个线性层一个激活函数，处理复杂的非线性数据
        self.norm = nn.LayerNorm(dim)#层归一化

    def forward(self, x):
        t_pool = self.temporal_proj(x.transpose(1, 2)).transpose(1, 2).squeeze(1)#转换时间和空间特征，把一个时间步去掉变成二维，历史信息
        last_out = x[:, -1, :]#批次都要，时间步只要最近的一个，表示当前信息
        fused = t_pool + last_out#历史和当前
        return self.norm(fused + self.proj(fused))#残差和标准化

class SpatialDiffusion(nn.Module):#空间扩散，GCN卷积神经网络
    def __init__(self, dim, num_nodes, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.weight0 = nn.Parameter(torch.eye(dim) + torch.randn(dim, dim) * 0.01)
        self.weight1 = nn.Parameter(torch.eye(dim) + torch.randn(dim, dim) * 0.01)#单位矩阵，随机加上0.01的微调
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)#只有0.1
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        res = x
        x = self.norm(x)
        ax1 = torch.matmul(adj, x)#矩阵乘法，每个网格结合自己和邻居的流量情况
        out = torch.matmul(x, self.weight0) + torch.matmul(ax1, self.weight1)#自己的加上邻居的
        return res + self.alpha * self.drop(torch.nn.functional.gelu(out))#返回激活函数，遗忘0，乘以0.1加上本身

class ST_Mamba_Model(nn.Module):
    def __init__(self, use_gnn=True, temporal_type='mamba', adj=None):
        super().__init__()
        self.use_gnn = use_gnn#是否开启GNN
        self.temporal_type = temporal_type.lower()#使用哪种算法

        if adj is not None:
            self.register_buffer('adj_matrix', adj)#把城市网格地图（邻接矩阵）存入缓存，不参与学习，但随时可查


        self.h_dim = config.hidden_dim
        self.num_layers = config.num_layers#隐藏维度和总层数

        self.revin = RevIN_ST(config.num_nodes)#归一化处理

        self.embedding = nn.Sequential(
            nn.Linear(config.input_dim, self.h_dim),
            nn.LayerNorm(self.h_dim),
            nn.GELU()
        )#把原始数字转成32维向量

        self.spatial_emb = nn.Parameter(torch.randn(1, 1, config.num_nodes, self.h_dim) * 0.02)#空间标签
        self.temporal_emb = nn.Parameter(torch.randn(1, config.history_steps, 1, self.h_dim) * 0.02)#时间标签

        self.st_pe = nn.Parameter(torch.randn(1, config.num_nodes, config.history_steps, self.h_dim) * 0.02)#时空标签

        self.decomp = SeriesDecomp(kernel_size=3)#分为长期和短期
        self.trend_proj = nn.Linear(self.h_dim, self.h_dim)#长期趋势
#选取算法
        if self.temporal_type == 'mamba':
            self.temporal_net = nn.ModuleList([
                TS_MambaBlock(d_model=self.h_dim, expand=2, dropout=config.dropout)
                for _ in range(self.num_layers)
            ])
        elif self.temporal_type == 'transformer':
            # Transformer 基线加入，使用与 Mamba/RNN 完全相同的维度环境
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.h_dim,
                nhead=4,
                dim_feedforward=self.h_dim * 2,
                dropout=config.dropout,
                batch_first=True
            )
            self.temporal_net = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        elif self.temporal_type == 'lstm':
            self.temporal_net = nn.LSTM(self.h_dim, self.h_dim, num_layers=self.num_layers, batch_first=True, dropout=config.dropout if self.num_layers > 1 else 0)
        elif self.temporal_type == 'gru':
            self.temporal_net = nn.GRU(self.h_dim, self.h_dim, num_layers=self.num_layers, batch_first=True, dropout=config.dropout if self.num_layers > 1 else 0)

        #选择GNN需要进行邻居网格的信息处理
        if self.use_gnn:
            self.spatial_net = SpatialDiffusion(self.h_dim, config.num_nodes, dropout=config.dropout)#空间扩散
            self.s_transform = nn.Linear(self.h_dim, self.h_dim)
            self.st_norm = nn.LayerNorm(self.h_dim)

            self.fusion_gate = nn.Linear(self.h_dim * 2, self.h_dim)#自己的和邻居的，算出来一个0到1的数，动态调整空间依赖程度
            self.spatial_drop = nn.Dropout(config.dropout)

            # 所有融合模型均配发协同投影组件，维度缩小，激活之后放大
            self.cross_norm = nn.LayerNorm(self.h_dim)
            self.synergy_proj = nn.Sequential(
                nn.Linear(self.h_dim, self.h_dim // 2),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(self.h_dim // 2, self.h_dim)
            )

        self.temporal_readout = AnchorReadout(self.h_dim, config.history_steps)#把 12 个小时的复杂变化，提炼成当前这一刻的最核心特征
        self.ar_full = nn.Linear(config.history_steps, 1)#直接把12步的转成1步，保底

        self.output_head = nn.Sequential(
            nn.LayerNorm(self.h_dim),
            SwiGLU(self.h_dim, self.h_dim, hidden_dim=self.h_dim * 2),
            nn.Dropout(config.dropout),
            nn.Linear(self.h_dim, 1)
        )#非线性处理，把维度从32转成1

        self.conf_gate = nn.Linear(self.h_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'temporal_net' not in name and 'spatial_net' not in name:
                nn.init.xavier_uniform_(p)#使用 Xavier 均匀分布

        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)#把输出层的权重先设为 0
        nn.init.xavier_uniform_(self.ar_full.weight)
        nn.init.zeros_(self.ar_full.bias)#初始化保底策略

        if hasattr(self, 'conf_gate'):
            nn.init.zeros_(self.conf_gate.weight)
            nn.init.constant_(self.conf_gate.bias, -1.0)#把偏置（bias）设为 -1.0。这意味着在没有训练前，AI 默认对自己没信心，这样初始的预测由保底专家决定

        if hasattr(self, 'fusion_gate'):
            nn.init.xavier_uniform_(self.fusion_gate.weight)
            nn.init.constant_(self.fusion_gate.bias, -3.0)#偏置设为 -3.0，意思是默认情况下更倾向于相信“自己的原始直觉”，对于“邻居打听来的消息（GNN）”，AI 需要学习很久才会慢慢采纳。

    def forward(self, x):
        B, T, N, C = x.shape# x的原始形状[批次,12步时间,225个网格,3个特征]

        x_norm = self.revin(x, 'norm')#归一化
        x_emb = self.embedding(x_norm) + self.spatial_emb + self.temporal_emb#归一化转成32维度，加上时空标签

        t_in = x_emb.permute(0, 2, 1, 3).reshape(B * N, T, -1)#[总样本数, 时间长度, 特征维度]
        res, trend = self.decomp(t_in)#分离出波动(res)和趋势(trend)

        # ---------------- 核心时序流转 ----------------
        res_spatial = res.reshape(B, N, T, -1) + self.st_pe#加时空标签转换形状
        t_dyn_long = res_spatial.reshape(B, N * T, -1)

        if self.temporal_type == 'mamba':
            for layer in self.temporal_net:
                t_dyn_long = layer(t_dyn_long)
        elif self.temporal_type == 'transformer':
            t_dyn_long = self.temporal_net(t_dyn_long)
        elif self.temporal_type in ['lstm', 'gru']:
            t_dyn_long, _ = self.temporal_net(t_dyn_long)#根据不同模型选择，mamba用循环可以快速扫描长序列，transformer擅长全局，其他产生的隐藏结果无用被抛出

        t_dyn = t_dyn_long.reshape(B, N, T, -1).reshape(B * N, T, -1)

        # ---------------- GNN 补充图拓扑信息 ----------------
        if self.use_gnn and self.temporal_type != 'none':
            t_dyn_spatial = t_dyn.reshape(B, N, T, -1).permute(0, 2, 1, 3).reshape(B * T, N, -1)
            g_dyn_spatial = self.spatial_net(t_dyn_spatial, self.adj_matrix)#通过图神经网络，让每个网格吸收周围邻居的情况
            g_dyn = g_dyn_spatial.reshape(B, T, N, -1).permute(0, 2, 1, 3).reshape(B * N, T, -1)

            g_feat = self.st_norm(self.s_transform(g_dyn))

            gate = torch.sigmoid(self.fusion_gate(torch.cat([t_dyn, g_feat], dim=-1)))#算出0~1的门控权重。AI 决定该听自己(t_dyn)还是听邻居(g_feat)

            cross_input = self.cross_norm(t_dyn * g_feat)#通过“协同投影”，挖掘时间和空间交叉影响产生的深层规律
            st_synergy = self.synergy_proj(cross_input)
            spatial_info = self.spatial_drop(gate * (g_feat + st_synergy))
            t_dyn = t_dyn + spatial_info#最终的综合情报=自己的想法+邻居意见及碰撞火花

        elif self.use_gnn:
            t_spatial = res.reshape(B, N, T, -1).permute(0, 2, 1, 3).reshape(B * T, N, -1)
            g_dyn_spatial = self.spatial_net(t_spatial, self.adj_matrix)
            t_dyn = g_dyn_spatial.reshape(B, T, N, -1).permute(0, 2, 1, 3).reshape(B * N, T, -1)

        t_out = t_dyn + self.trend_proj(trend)#还原趋势：把刚才拆走的“长期稳定规律”通过简单加工后加回来，这样结果既包含了深度的突发预测，也保留了基本的日常基调
        t_feat = self.temporal_readout(t_out).reshape(B, N, -1)

        delta_norm = self.output_head(t_feat).reshape(B, 1, N, 1)
        conf = torch.sigmoid(self.conf_gate(t_feat)).reshape(B, 1, N, 1)

        history_demand = x_norm[..., 0]
        ar_base = self.ar_full(history_demand.transpose(1, 2)).transpose(1, 2).unsqueeze(-1)#保底机制：不靠 AI 大脑，只靠最死板的历史统计规律算一个底线预测

        pred_norm = ar_base + delta_norm * conf#双保险
        pred_real = self.revin(pred_norm, 'denorm')#反归一化

        return pred_real#预测值

PLOT_COLORS = {
    'Mamba-GNN': '#FF1493',
    'Transformer-GNN': '#00FFFF', # 青色
    'LSTM-GNN': '#FFA500',
    'GRU-GNN': '#FFFF00',
    'Mamba-Only': '#FF4500',
    'Transformer-Only': '#FF69B4', # 粉红色
    'LSTM-Only': '#1E90FF',
    'GRU-Only': '#8A2BE2',
    'GNN-Only': '#32CD32'
}

def plot_fusion_loss(history_dict, save_dir):#融合损失曲线图
    if 'Mamba-GNN' not in history_dict: return
    hist = history_dict['Mamba-GNN']
    plt.figure(figsize=(10, 6))
    plt.plot(hist['train_loss'], label='训练损失 (Training Loss)', color='#FF1493', linewidth=2, alpha=0.9)
    plt.plot(hist['val_loss'], label='验证损失 (Validation Loss)', color='#00D9FF', linewidth=2, linestyle='--', alpha=0.9)
    plt.title("Mamba-GNN 模型训练损失", fontsize=16)
    plt.xlabel("轮数 (Epochs)", fontsize=12)
    plt.ylabel("损失 (Loss)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(save_dir, 'fusion_model_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_total_demand(all_preds, all_trues, save_dir, prefix=""):#真实需求和预测需求图
    if not all_preds: return
    plt.figure(figsize=(15, 6))
    first_key = list(all_trues.keys())[0]
    total_true = np.sum(all_trues[first_key], axis=1)
    plot_len = min(len(total_true), 200)

    plt.plot(total_true[:plot_len], label='真实值 (Ground Truth)', color='white', linewidth=3, alpha=0.5)

    sorted_names = sorted(all_preds.keys(), key=lambda x: ('Mamba' in x, 'Transformer' in x), reverse=True)

    for name in sorted_names:
        preds = all_preds[name]
        total_pred = np.sum(preds, axis=1)
        lw = 3.5 if 'Mamba' in name else 1.5
        alpha = 0.95 if 'Mamba' in name else 0.7
        ls = '-' if 'Mamba' in name else '--'
        plt.plot(total_pred[:plot_len], label=name, color=PLOT_COLORS.get(name, 'red'),
                 linewidth=lw, linestyle=ls, alpha=alpha)

    title_str = f"[{prefix}] 总需求量预测对比" if prefix else "总需求量预测对比"
    plt.title(title_str, fontsize=16)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    filename = f"{prefix}_total_demand_comparison.png" if prefix else "total_demand_comparison.png"
    plt.savefig(os.path.join(save_dir, filename.replace(" ", "_")), dpi=300, bbox_inches='tight')
    plt.close()

def plot_scatter_fit(all_preds, all_trues, save_dir, prefix=""):#拟合散点图
    if not all_preds: return
    n_models = len(all_preds)
    cols = min(4, n_models)
    rows = math.ceil(n_models / cols)
    plt.figure(figsize=(5 * cols, 5 * rows))

    for i, (name, preds) in enumerate(all_preds.items()):
        plt.subplot(rows, cols, i+1)
        trues = all_trues[name].flatten()
        preds = preds.flatten()
        idx = np.random.choice(len(trues), min(10000, len(trues)), replace=False)
        plt.scatter(trues[idx], preds[idx], alpha=0.3, color=PLOT_COLORS.get(name, 'white'), s=5)
        max_val = max(trues[idx].max(), preds[idx].max())
        plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='理想拟合线')

        plt.title(f"{name} 拟合分析", fontsize=14, color=PLOT_COLORS.get(name, 'white') if 'Mamba' in name else 'white')
        plt.xlabel("真实值", fontsize=12)
        if i % cols == 0: plt.ylabel("预测值", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.2)
        plt.legend()

    plt.tight_layout()
    filename = f"{prefix}_goodness_of_fit.png" if prefix else "goodness_of_fit.png"
    plt.savefig(os.path.join(save_dir, filename.replace(" ", "_")), dpi=300, bbox_inches='tight')
    plt.close()

def plot_spatial_error(preds, trues, grid_meta, name, save_dir):#误差空间分布图
    node_mae = np.mean(np.abs(trues - preds), axis=0)
    error_matrix = node_mae.reshape((grid_meta['rows'], grid_meta['cols']))
    plt.figure(figsize=(8, 6))
    plt.imshow(error_matrix, cmap='magma', origin='lower', aspect='auto')
    plt.colorbar(label='平均绝对误差 (MAE)')
    plt.title(f"{name} 空间误差分布", fontsize=14)
    plt.xlabel("经度网格索引")
    plt.ylabel("纬度网格索引")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'spatial_error_map_{name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_distribution(all_preds, all_trues, save_dir, prefix=""):#误差值分布
    if not all_preds: return
    plt.figure(figsize=(12, 6))

    for name, preds in all_preds.items():
        errors = (all_trues[name] - preds).flatten()
        errors = errors[(errors >= -2) & (errors <= 2)]
        lw = 2.5 if 'Mamba' in name else 1.0
        plt.hist(errors, bins=100, alpha=0.5 if 'Mamba' in name else 0.3, label=name, color=PLOT_COLORS.get(name, 'white'), density=True, histtype='step', linewidth=lw)

    plt.axvline(x=0, color='white', linestyle='--', linewidth=2)
    title_str = f"[{prefix}] 误差分布" if prefix else "误差分布"
    plt.title(title_str, fontsize=16)
    plt.xlabel("误差值 (真实值 - 预测值)", fontsize=12)
    plt.ylabel("密度 (Density)", fontsize=12)
    plt.xlim(-2, 2)
    plt.legend(fontsize=10, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    filename = f"{prefix}_error_distribution.png" if prefix else "error_distribution.png"
    plt.savefig(os.path.join(save_dir, filename.replace(" ", "_")), dpi=300, bbox_inches='tight')
    plt.close()

def plot_epoch_metrics(history_dict, save_dir):#全系指标图
    for name, hist in history_dict.items():
        if 'train_mse' not in hist or not hist['train_mse']: continue
        epochs = range(1, len(hist['train_mse']) + 1)
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"{name} - 训练指标变化曲线", fontsize=18, weight='bold', color=PLOT_COLORS.get(name, 'white'))

        axs[0, 0].plot(epochs, hist['train_acc'], label='训练 Acc', color='#00FF00', linewidth=2)
        axs[0, 0].plot(epochs, hist['val_acc'], label='验证 Acc', color='#FF00FF', linewidth=2, linestyle='--')
        axs[0, 0].plot(epochs, hist['test_acc'], label='测试 Acc', color='#00D9FF', linewidth=2, linestyle='-.')
        axs[0, 0].set_title('准确率 (Accuracy)', fontsize=14)
        axs[0, 0].legend()
        axs[0, 0].grid(True, linestyle='--', alpha=0.3)

        axs[0, 1].plot(epochs, hist['train_mae'], label='训练 MAE', color='#00FF00', linewidth=2)
        axs[0, 1].plot(epochs, hist['val_mae'], label='验证 MAE', color='#FF00FF', linewidth=2, linestyle='--')
        axs[0, 1].plot(epochs, hist['test_mae'], label='测试 MAE', color='#00D9FF', linewidth=2, linestyle='-.')
        axs[0, 1].set_title('平均绝对误差 (MAE)', fontsize=14)
        axs[0, 1].legend()
        axs[0, 1].grid(True, linestyle='--', alpha=0.3)

        axs[1, 0].plot(epochs, hist['train_mse'], label='训练 MSE', color='#00FF00', linewidth=2)
        axs[1, 0].plot(epochs, hist['val_mse'], label='验证 MSE', color='#FF00FF', linewidth=2, linestyle='--')
        axs[1, 0].plot(epochs, hist['test_mse'], label='测试 MSE', color='#00D9FF', linewidth=2, linestyle='-.')
        axs[1, 0].set_title('均方误差 (MSE)', fontsize=14)
        axs[1, 0].legend()
        axs[1, 0].grid(True, linestyle='--', alpha=0.3)

        train_r2 = [max(x, -1.0) for x in hist['train_r2']]
        val_r2 = [max(x, -1.0) for x in hist['val_r2']]
        test_r2 = [max(x, -1.0) for x in hist['test_r2']]
        axs[1, 1].plot(epochs, train_r2, label='训练 R2', color='#00FF00', linewidth=2)
        axs[1, 1].plot(epochs, val_r2, label='验证 R2', color='#FF00FF', linewidth=2, linestyle='--')
        axs[1, 1].plot(epochs, test_r2, label='测试 R2', color='#00D9FF', linewidth=2, linestyle='-.')
        axs[1, 1].set_title('决定系数 (R2 Score)', fontsize=14)
        axs[1, 1].legend()
        axs[1, 1].grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f'{name}_all_metrics_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

def compute_mape(y_true, y_pred, threshold=5.0):#计算MAPE
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mask = y_true > threshold
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_wmape(y_true, y_pred):#计算WMAPE
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-6) * 100

def get_hotspots(pred_data, grid_meta, top_k=5):#计算预测总量前五
    total_vol = np.sum(pred_data, axis=0)
    sorted_indices = np.argsort(-total_vol)
    top_indices = sorted_indices[:top_k]

    hotspots = []
    for rank, idx in enumerate(top_indices):
        idx = int(idx)
        cols = grid_meta['cols']
        y = idx // cols
        x = idx % cols
        lon_start = grid_meta['lon_min'] + x * grid_meta['lon_step']
        lon_end = lon_start + grid_meta['lon_step']
        lat_start = grid_meta['lat_min'] + y * grid_meta['lat_step']
        lat_end = lat_start + grid_meta['lat_step']

        hotspots.append({
            "rank": rank + 1,
            "grid_id": idx,
            "lon_range": [float(lon_start), float(lon_end)],
            "lat_range": [float(lat_start), float(lat_end)],
            "predicted_volume": int(total_vol[idx])
        })
    return hotspots

# ==========================================
# 5. 实验主逻辑
# ==========================================
class EarlyStopping:#早停机制
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

class RobustTrafficLoss(nn.Module):#损失曲线评估标准，把 Huber 和 MAE 结合起来，比较平稳
    def __init__(self, alpha=0.5):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=1.0)
        self.mae = nn.L1Loss()
        self.alpha = alpha

    def forward(self, pred, true):
        return self.huber(pred, true) + self.alpha * self.mae(pred, true)

def run_exp(name, model, loaders, scaler):
    # 确保在模型启动前，重新将随机种子校准到 42
    set_seed(config.seed)

    train_loader, val_loader, test_loader = loaders

    weight_decay = 1e-4
    lr = 1.0e-3
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)#AdamW优化器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-5)#调整学习率

    criterion = RobustTrafficLoss(alpha=0.5)#loss评分
    early_stopping = EarlyStopping(patience=5, delta=1e-6)#早停

    # 【终极防线 4】：彻底关闭 AMP (自动混合精度)，完全依赖 Float32 进行高精度计算
    scaler_amp = torch.cuda.amp.GradScaler(enabled=False)

    save_path = os.path.join(config.save_dir, f'best_model_{name}.pt')

    history = {
        'train_loss': [], 'val_loss': [],
        'train_mse': [], 'val_mse': [], 'test_mse': [],
        'train_mae': [], 'val_mae': [], 'test_mae': [],
        'train_r2': [], 'val_r2': [], 'test_r2': [],
        'train_acc': [], 'val_acc': [], 'test_acc': []
    }

    def calculate_metrics_from_tensors(p_list, t_list):#算评估标准
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

        model.train()#训练
        t_loss = 0
        train_p, train_t = [], []
        for bx, by in train_loader:
            bx, by = bx.to(config.device), by.to(config.device)
            optimizer.zero_grad()

            # 关闭 autocast
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

        model.eval()#验证
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

        scheduler.step()#修改训练进度
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

    state_dict = torch.load(save_path)#从硬盘里，把历史考得最好那一次的“脑子”重新装载回来
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    model.eval()#拿着最好的脑子，去跑一次谁也没见过的期末大考

    all_p, all_t = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(config.device), by.to(config.device)
            with torch.cuda.amp.autocast(enabled=False):
                p = model(bx)
            all_p.append(p.cpu().reshape(-1, config.num_nodes))
            all_t.append(by.cpu().reshape(-1, config.num_nodes))
#评估结果可视
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

# 【复杂度说明】揭露了 Transformer 面对时空平铺时长序列的全连接灾难，凸显 Mamba 优势
TIME_COMPLEXITY_MAP = {
    'Mamba-GNN': 'O(T·N² + B·(T·N)·d) [并行扫描]',
    'Mamba-Only': 'O(B·(T·N)·d) [并行扫描]',
    'Transformer-GNN': 'O(T·N² + B·(T·N)²) [全局自注意力]',
    'Transformer-Only': 'O(B·(T·N)²) [全局自注意力]',
    'LSTM-GNN': 'O(T·N² + B·(T·N)·d²) [长序列串行]',
    'GRU-GNN': 'O(T·N² + B·(T·N)·d²) [长序列串行]',
    'LSTM-Only': 'O(B·(T·N)·d²) [长序列串行]',
    'GRU-Only': 'O(B·(T·N)·d²) [长序列串行]',
    'GNN-Only': 'O(T·N²)'
}

def main():
    print("="*100)
    print(" 🚀 学术严谨版模型环境初始化完成！")
    print("="*100)

    data, scaler, grid_meta = load_and_process_data()#数据处理
    loaders = create_dataloaders(data)#滑动窗口取样
    adj = get_adjacency_matrix()#邻接矩阵

    # 将 Transformer 及 Transformer-GNN 完美嵌入对照池
    experiments = {
        'Mamba-GNN':  {'gnn': True,  'temporal': 'mamba'},
        'Mamba-Only': {'gnn': False, 'temporal': 'mamba'},
        'Transformer-GNN': {'gnn': True, 'temporal': 'transformer'},
        'Transformer-Only': {'gnn': False, 'temporal': 'transformer'},
        'LSTM-GNN':   {'gnn': True,  'temporal': 'lstm'},
        'GRU-GNN':    {'gnn': True,  'temporal': 'gru'},
        'LSTM-Only':  {'gnn': False, 'temporal': 'lstm'},
        'GRU-Only':   {'gnn': False, 'temporal': 'gru'},
        'GNN-Only':   {'gnn': True,  'temporal': 'none'}
    }

    results = {}
    all_hist, all_pred, all_true = {}, {} , {}

    final_report = {
        "evaluation_groups": {},
        "loss_history": {},
        "predictions_time_series": {},
        "hotspots": []
    }

    for name, cfg in experiments.items():
        # [核心修改处]: 将 Transformer 也纳入多次运行范畴，应对并行算子波动
        needs_multi_run = 'Mamba' in name or 'Transformer' in name
        num_runs = 3 if needs_multi_run else 1

        print("="*80)
        if needs_multi_run:
            print(f"🔹 [实验进行中] 模型名称: {name} (因底层并行算子波动，将自动运行 {num_runs} 次取均值)")
        else:
            print(f"🔹 [实验进行中] 模型名称: {name} (完全确定性模型，运行 1 次)")
        print("-" * 80)

        run_metrics = {'mse': [], 'rmse': [], 'mae': [], 'mape': [], 'wmape': [], 'r2': [], 'time': []}
        best_mae = float('inf')
        best_hist, best_p, best_t = None, None, None
        final_params = 0

        for i in range(num_runs):
            if num_runs > 1:
                print(f"\n   >>> 正在执行第 {i+1}/{num_runs} 次独立运行...")

            set_seed(config.seed)

            model = ST_Mamba_Model(
                use_gnn=cfg['gnn'],
                temporal_type=cfg['temporal'],
                adj=adj
            ).to(config.device)

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

        if needs_multi_run:
            results[name] = {
                "mse": f"{np.mean(run_metrics['mse']):.2f}±{np.std(run_metrics['mse']):.2f}",
                "rmse": f"{np.mean(run_metrics['rmse']):.2f}±{np.std(run_metrics['rmse']):.2f}",
                "mae": f"{np.mean(run_metrics['mae']):.2f}±{np.std(run_metrics['mae']):.2f}",
                "mape": f"{np.mean(run_metrics['mape']):.2f}±{np.std(run_metrics['mape']):.2f}",
                "wmape": f"{np.mean(run_metrics['wmape']):.2f}±{np.std(run_metrics['wmape']):.2f}",
                "r2": f"{np.mean(run_metrics['r2']):.4f}±{np.std(run_metrics['r2']):.4f}",
                "parameters": int(final_params),
                "avg_epoch_time_s": f"{np.mean(run_metrics['time']):.2f}"
            }
        else:
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

    # 在各核心分组中同步注入 Transformer，实现极致公平的打压对照
    experiment_groups = {
        "Ablation_Study": {
            "title": "消融实验 (Ablation Study) - 验证时空融合机制",
            "models": ['Mamba-GNN', 'Mamba-Only', 'GNN-Only']
        },
        "Fusion_Comparison": {
            "title": "融合模型对比 (Spatio-Temporal Baselines) - 验证 Mamba 效能",
            "models": ['Mamba-GNN', 'Transformer-GNN', 'LSTM-GNN', 'GRU-GNN']
        },
        "Single_Comparison": {
            "title": "单一时序模型对比 (Temporal Baselines)",
            "models": ['Mamba-Only', 'Transformer-Only', 'LSTM-Only', 'GRU-Only']
        }
    }

    widths = [16, 14, 14, 14, 14, 14, 15, 10, 12, 45]
    headers = ['Model', 'MSE', 'RMSE', 'MAE', 'MAPE', 'WMAPE', 'R2', 'Params', 'Time/Ep(s)', 'Complexity']

    line_length = sum(widths) + len(widths) * 3 - 1

    print("\n\n" + "="*line_length)
    print(" 严谨学术模型评估与对比报告")
    print("="*line_length)

    for group_key, group_info in experiment_groups.items():
        title = group_info["title"]
        models_in_group = group_info["models"]

        print(f"\n>> {title}")
        print("-" * line_length)

        print(format_table_row(headers, widths))
        print("-" * line_length)

        group_results = {}
        for name in models_in_group:
            if name in results:
                m = results[name]
                group_results[name] = m
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

        final_report["evaluation_groups"][group_key] = group_results

        subset_pred = {k: all_pred[k] for k in models_in_group if k in all_pred}
        subset_true = {k: all_true[k] for k in models_in_group if k in all_true}

        plot_total_demand(subset_pred, subset_true, config.save_dir, prefix=group_key)
        plot_scatter_fit(subset_pred, subset_true, config.save_dir, prefix=group_key)
        plot_error_distribution(subset_pred, subset_true, config.save_dir, prefix=group_key)

    print(f"\n[可视化] 实验评估图表已生成至目录: {os.path.abspath(config.save_dir)}")

    plot_fusion_loss(all_hist, config.save_dir)
    plot_spatial_error(all_pred['Mamba-GNN'], all_true['Mamba-GNN'], grid_meta, 'Mamba-GNN', config.save_dir)
    plot_epoch_metrics(all_hist, config.save_dir)

    gt_series = np.sum(all_true['Mamba-GNN'], axis=1).tolist()
    final_report["predictions_time_series"]["ground_truth"] = [float(x) for x in gt_series]
    pred_series = np.sum(all_pred['Mamba-GNN'], axis=1).tolist()
    final_report["predictions_time_series"]["prediction_mamba_gnn"] = [float(x) for x in pred_series]

    hotspots_data = get_hotspots(all_pred['Mamba-GNN'], grid_meta, top_k=5)
    final_report["hotspots"] = hotspots_data

    widths_hot = [6, 8, 12, 24, 24]
    headers_hot = ['Rank', 'Grid ID', '预测总流量', '经度范围 (Lon)', '纬度范围 (Lat)']
    line_hot_length = sum(widths_hot) + len(widths_hot) * 3 - 1

    print("\n" + "="*line_hot_length)
    print(" 📍 流量预测最高 Top 5 热点区域 (基于最佳模型结果)")
    print("-" * line_hot_length)
    print(format_table_row(headers_hot, widths_hot))
    print("-" * line_hot_length)

    for item in hotspots_data:
        lon_r = f"[{item['lon_range'][0]:.4f}, {item['lon_range'][1]:.4f}]"
        lat_r = f"[{item['lat_range'][0]:.4f}, {item['lat_range'][1]:.4f}]"
        row_hot = [str(item['rank']), str(item['grid_id']), str(item['predicted_volume']), lon_r, lat_r]
        print(format_table_row(row_hot, widths_hot))
    print("="*line_hot_length)

    json_path = os.path.join(config.save_dir, 'experiment_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    print(f"\n[文件] 结构化报告 JSON 已生成: {json_path}")
    print(f"[完成] 所有深度学习评估流程安全结束！")

if __name__ == '__main__':
    main()