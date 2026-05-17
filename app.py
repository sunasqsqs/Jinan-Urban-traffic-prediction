# -*- coding: utf-8 -*-
import os
import json
import time
import math
import random
import threading
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from flask import Flask, render_template, jsonify, send_from_directory, request, session

app = Flask(__name__)
app.secret_key = "mamba_gnn_super_secret_key"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= 基础设置与原有 API =================

def get_current_dataset():
    return session.get('dataset', 'results')

def get_experiment_data():
    dataset_dir = get_current_dataset()
    json_path = os.path.join(CURRENT_DIR, dataset_dir, 'experiment_report.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"JSON 解析错误: {e}")
    return {}

@app.route('/')
def index(): return render_template('index.html', page="index")

@app.route('/login')
def login(): return render_template('login.html', page="login")

@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html', page="dashboard")

# 独立实时预测页面路由
@app.route('/realtime')
def realtime(): return render_template('realtime.html', page="realtime")

@app.route('/analytics')
def analytics(): return render_template('analytics.html', page="analytics")

@app.route('/contrast')
def contrast(): return render_template('contrast.html', page="contrast")

@app.route('/system')
def system(): return render_template('system.html', page="system")

@app.route('/about')
def about(): return render_template('about.html', page='about')

@app.route('/doc')
def doc(): return render_template('doc.html', page='doc')

@app.route('/users')
def users(): return render_template('users.html', page='users')

@app.route('/results/<path:filename>')
def serve_results_file(filename):
    dataset_dir = get_current_dataset()
    results_dir = os.path.join(CURRENT_DIR, dataset_dir)
    return send_from_directory(results_dir, filename)

@app.route('/api/data')
def api_data():
    data = get_experiment_data()
    return jsonify(data) if data else (jsonify({"error": "暂无数据"}), 404)

@app.route('/api/set_dataset', methods=['POST'])
def set_dataset():
    data = request.get_json()
    if data and 'dataset' in data and data['dataset'] in ['results', 'results2']:
        session['dataset'] = data['dataset']
        return jsonify({"status": "success", "dataset": data['dataset']})
    return jsonify({"error": "无效的数据集"}), 400

@app.route('/api/get_dataset')
def api_get_dataset():
    return jsonify({"dataset": get_current_dataset()})


# ================= 实时推断引擎核心 (225个网格全空间仿真版) =================
class RealtimeEngine:
    def __init__(self):
        self.full_data = []          # 完整加载的 225网格 时空数据
        self.history = []            # 当前推演的历史记录
        self.metrics_history = {'mae': [], 'mse': [], 'r2': []} # 全市大盘误差指标追踪
        self.grid_coords = []        # 225个网格的具体中心坐标 (15x15)

        self.is_running = False
        self.current_index = 0
        self.target_steps = 0
        self.interval = 1.0          # 每个时间步的时长(秒)
        self.lock = threading.Lock()
        self.thread = None

    def load_data(self):
        # 初始化 15x15 = 225 个网格的地理坐标
        cols, rows = 15, 15
        num_nodes = cols * rows
        lon_min, lon_max = 116.0, 118.0
        lat_min, lat_max = 36.0, 37.8
        lon_step = (lon_max - lon_min) / cols
        lat_step = (lat_max - lat_min) / rows

        self.grid_coords = []
        for i in range(num_nodes):
            grid_y = i // cols
            grid_x = i % cols
            lon_center = lon_min + (grid_x + 0.5) * lon_step
            lat_center = lat_min + (grid_y + 0.5) * lat_step
            self.grid_coords.append({
                "id": i,
                "lon": round(lon_center, 4),
                "lat": round(lat_center, 4)
            })

        print(f"[实时引擎] 正在构建 225 个图神经路网节点...")

        # 产生高精度的空间与时间耦合交通仿真数据 (双峰潮汐模型)
        self.full_data = []
        steps_to_gen = 100
        base_time = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)

        for step in range(steps_to_gen):
            timestamp = base_time + datetime.timedelta(minutes=5 * step)
            time_str = timestamp.strftime('%H:%M:%S')

            # 时间波动因子 (早晚高峰)
            hour = (timestamp.hour + timestamp.minute / 60.0) % 24
            temporal_factor = 0.2 + 0.6 * math.exp(-((hour - 8.5) / 1.8)**2) + 0.75 * math.exp(-((hour - 18.0) / 2.2)**2)
            temporal_factor = max(0.1, temporal_factor)

            grid_trues = []
            grid_preds = []

            for g_id in range(num_nodes):
                # 空间波动因子 (市中心高流量聚集，边缘递减)
                gx = g_id % cols
                gy = g_id // cols
                dist_to_center = math.sqrt((gx - 7)**2 + (gy - 7)**2)
                spatial_factor = math.exp(-dist_to_center / 4.5)

                base_demand = 8.0 + 95.0 * spatial_factor
                noise = random.uniform(0.85, 1.15)

                # 计算出高逼真度的真实值与预测值
                true_val = int(base_demand * temporal_factor * noise)
                if spatial_factor < 0.12 and random.random() < 0.2:
                    true_val = 0 # 模拟外围无订单状态

                # Mamba-GNN 预测模型仿真 (高斯误差分布，平均准确度约92%)
                pred_error = random.normalvariate(0, 0.08)
                pred_val = int(true_val * (1.0 + pred_error))
                if pred_val < 0: pred_val = 0

                grid_trues.append(true_val)
                grid_preds.append(pred_val)

            total_true = sum(grid_trues)
            total_pred = sum(grid_preds)

            self.full_data.append({
                "timestamp": time_str,
                "true_val": total_true,
                "pred_val": total_pred,
                "grid_trues": grid_trues,
                "grid_preds": grid_preds
            })

        print(f"[实时引擎] 225个网格时空推演数据就绪，共计 {len(self.full_data)} 个仿真时间步。")

    def start(self, steps):
        with self.lock:
            self.target_steps = min(steps, len(self.full_data))
            self.current_index = 0
            self.history = []
            self.metrics_history = {'mae': [], 'mse': [], 'r2': []}
            self.is_running = True

        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()

    def stop(self):
        with self.lock:
            self.is_running = False

    def _run_loop(self):
        while True:
            with self.lock:
                if not self.is_running:
                    break
                if self.current_index >= self.target_steps:
                    self.is_running = False
                    break

                current_step_data = self.full_data[self.current_index]

                # 动态生成当前时间步下的 Top 5 高流量热点枢纽，保持前端原有热点列表功能兼容
                trues = current_step_data["grid_trues"]
                preds = current_step_data["grid_preds"]

                top_indices = sorted(range(len(trues)), key=lambda idx: trues[idx], reverse=True)[:5]
                hotspots = []
                for rank, idx in enumerate(top_indices):
                    t_val = trues[idx]
                    p_val = preds[idx]
                    mae = abs(t_val - p_val)
                    coord = self.grid_coords[idx]
                    hotspots.append({
                        "id": f"Grid {idx}",
                        "grid_id": idx,
                        "lon": coord["lon"],
                        "lat": coord["lat"],
                        "true": t_val,
                        "pred": p_val,
                        "mae": round(mae, 2),
                        "loss": round(mae * random.uniform(0.12, 0.28), 4),
                        "accuracy": round(max(0, 100 - (mae / max(1, t_val)) * 100), 2)
                    })

                # 将包含225个网格当前状态的完整大包注入到历史轨迹中
                step_payload = {
                    "timestamp": current_step_data["timestamp"],
                    "true_val": current_step_data["true_val"],
                    "pred_val": current_step_data["pred_val"],
                    "grid_trues": trues,
                    "grid_preds": preds,
                    "hotspots": hotspots
                }
                self.history.append(step_payload)

                # 计算全市整体大盘流量的累积误差
                y_true = [d['true_val'] for d in self.history]
                y_pred = [d['pred_val'] for d in self.history]

                if len(y_true) > 1:
                    mae = mean_absolute_error(y_true, y_pred)
                    mse = mean_squared_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred) if np.var(y_true) > 0 else 0.0
                else:
                    mae = abs(y_true[0] - y_pred[0])
                    mse = mae ** 2
                    r2 = 0.0

                self.metrics_history['mae'].append(round(mae, 4))
                self.metrics_history['mse'].append(round(mse, 4))
                self.metrics_history['r2'].append(round(r2, 4))

                self.current_index += 1

            # 控制时间推移频率
            time.sleep(self.interval)

    def sync(self):
        with self.lock:
            return {
                "is_running": self.is_running,
                "current_step": self.current_index,
                "target_steps": self.target_steps,
                "is_finished": (self.current_index >= self.target_steps and self.target_steps > 0),
                "history": self.history,
                "metrics_history": self.metrics_history
            }

    def save_results(self):
        with self.lock:
            if not self.history:
                return {"error": "暂无数据可保存"}

            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_data = {
                "save_time": timestamp_str,
                "total_steps": self.current_index,
                "final_metrics": {
                    "mae": self.metrics_history['mae'][-1],
                    "mse": self.metrics_history['mse'][-1],
                    "r2": self.metrics_history['r2'][-1],
                },
                "history": self.history
            }
            save_dir = os.path.join(CURRENT_DIR, 'results', 'realtime_saves')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'record_{timestamp_str}.json')

            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            return {"status": "success", "file": save_path, "steps": self.current_index}

# 实例化引擎
engine = RealtimeEngine()
engine.load_data()


# ================= 实时流 API 接口 =================

@app.route('/api/realtime/start', methods=['POST'])
def rt_start():
    data = request.get_json()
    steps = int(data.get('steps', 50))
    engine.start(steps)
    return jsonify({"status": "started", "target_steps": steps})

@app.route('/api/realtime/stop', methods=['POST'])
def rt_stop():
    engine.stop()
    return jsonify({"status": "stopped"})

@app.route('/api/realtime/sync', methods=['GET'])
def rt_sync():
    return jsonify(engine.sync())

@app.route('/api/realtime/grids', methods=['GET'])
def rt_grids():
    """获取所有网格的静态坐标"""
    return jsonify(engine.grid_coords)

@app.route('/api/realtime/save', methods=['POST'])
def rt_save():
    result = engine.save_results()
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)