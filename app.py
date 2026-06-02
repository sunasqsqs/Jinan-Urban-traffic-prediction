# -*- coding: utf-8 -*-
import os
import json
import time
import math
import random
import threading
import datetime
import uuid
import urllib.parse
from threading import Lock
import numpy as np
import pandas as pd
from flask import Flask, render_template, render_template_string, jsonify, send_from_directory, request, session

app = Flask(__name__)
app.secret_key = "mamba_gnn_super_secret_key"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= 1. 线程安全打车模拟与调度指令内存数据库 =================
taxi_state_lock = Lock()
TAXI_DRIVERS = {}       # 存放所有在线司机 {driver_id: {name, location, status, vehicle, rating, user_type}}
TAXI_TRIPS = {}         # 存放所有打车行程 {trip_id: {passenger_id, passenger_name, pickup, dropoff, status, ...}}
SEED_PASSENGERS = {}    # 预置乘客账号 (服务器内存缓存)
DISPATCH_ORDERS = {}    # 网格调度广播订单 {order_id: {grid_id, needed, accepted, accepted_by[], status}}
LATEST_DISPATCH_ALERT = None  # 存放最新的智能调度派发指令
IGNORED_DISPATCH_ORDERS = {}  # 司机忽略的调度订单 {driver_id: set(order_ids)}

# 济南市核心商圈地标 (与高德地图高精度坐标完全绑定)
JINAN_LANDMARKS = [
    {"id": "quancheng_sq", "name": "泉城广场 (市中心)", "lng": 117.024967, "lat": 36.661156, "icon": "🏙️"},
    {"id": "baotu_spring", "name": "趵突泉景区", "lng": 117.015577, "lat": 36.660634, "icon": "⛲"},
    {"id": "daming_lake", "name": "大明湖风景区", "lng": 117.025354, "lat": 36.672957, "icon": "⛵"},
    {"id": "jinan_west", "name": "济南西站 (高铁站)", "lng": 116.886368, "lat": 36.668102, "icon": "🚄"},
    {"id": "yaoqiang_airport", "name": "遥墙国际机场", "lng": 117.265531, "lat": 36.850785, "icon": "✈️"},
    {"id": "qianfoshan", "name": "千佛山公园", "lng": 117.032649, "lat": 36.638531, "icon": "⛰️"},
    {"id": "gaoxin_wanda", "name": "高新区万达广场", "lng": 117.135235, "lat": 36.673891, "icon": "🛍️"},
]

VEHICLE_TYPES = [
    {"id": "sd_xuan", "name": "鲁A·优选新能源 (比亚迪秦)", "rate": 1.6, "desc": "经济环保，泉城绿色出行首选"},
    {"id": "sd_shushi", "name": "鲁A·舒适专车 (帕萨特)", "rate": 2.2, "desc": "空间宽敞，商务出行高品质"},
    {"id": "sd_luxury", "name": "鲁A·尊享豪华 (奥迪A6L)", "rate": 3.8, "desc": "高端座驾，尊贵礼遇与明星司机"}
]

def seed_ai_taxi_driver():
    """批量注册系统预置司机，分散在济南市各区，模拟真实运力分布"""
    SEED_DRIVERS = [
        {"id": "driver_zhang", "name": "张师傅", "lng": 117.024967, "lat": 36.661156, "vehicle": "鲁A·D88921 (特斯拉 Model Y)", "rating": 4.98},
        {"id": "driver_li",   "name": "李师傅", "lng": 117.015577, "lat": 36.660634, "vehicle": "鲁A·F12345 (比亚迪 秦)",    "rating": 4.85},
        {"id": "driver_wang", "name": "王师傅", "lng": 116.886368, "lat": 36.668102, "vehicle": "鲁A·G67890 (帕萨特)",        "rating": 4.92},
        {"id": "driver_zhao", "name": "赵师傅", "lng": 117.135235, "lat": 36.673891, "vehicle": "鲁A·H11223 (丰田 凯美瑞)",   "rating": 4.75},
        {"id": "driver_sun",  "name": "孙师傅", "lng": 117.032649, "lat": 36.638531, "vehicle": "鲁A·J33445 (大众 迈腾)",     "rating": 4.88},
        {"id": "driver_chen", "name": "陈师傅", "lng": 117.063889, "lat": 36.683333, "vehicle": "鲁A·K55667 (吉利 星瑞)",     "rating": 4.90},
        {"id": "driver_zhou", "name": "周师傅", "lng": 116.984722, "lat": 36.653333, "vehicle": "鲁A·L77889 (日产 轩逸)",     "rating": 4.81},
        {"id": "driver_wu",   "name": "吴师傅", "lng": 117.122500, "lat": 36.657222, "vehicle": "鲁A·M99001 (红旗 H5)",       "rating": 4.95},
        {"id": "driver_ma",   "name": "马师傅", "lng": 117.007500, "lat": 36.661389, "vehicle": "鲁A·N22334 (传祺 M8)",      "rating": 4.78},
        {"id": "driver_liu",  "name": "刘师傅", "lng": 117.098611, "lat": 36.653056, "vehicle": "鲁A·P44556 (蔚来 ES6)",     "rating": 4.86},
    ]

    with taxi_state_lock:
        for drv in SEED_DRIVERS:
            if drv["id"] not in TAXI_DRIVERS:
                TAXI_DRIVERS[drv["id"]] = {
                    "driver_id": drv["id"],
                    "name": drv["name"],
                    "location": {"lng": drv["lng"], "lat": drv["lat"]},
                    "status": "online",
                    "vehicle": drv["vehicle"],
                    "rating": drv["rating"],
                    "user_type": "driver",
                    "updated_at": time.time()
                }

# 预置乘客账号与订单种子数据
PASSENGER_SEED_DATA = [
    {"id": "pax_wang",  "name": "王先生", "phone": "138****6789", "pwd": "123456", "role": "normal", "user_type": "passenger"},
    {"id": "pax_li",    "name": "李女士", "phone": "139****8901", "pwd": "123456", "role": "normal", "user_type": "passenger"},
    {"id": "pax_zhao",  "name": "赵同学", "phone": "156****2345", "pwd": "123456", "role": "normal", "user_type": "passenger"},
    {"id": "pax_chen",  "name": "陈经理", "phone": "185****4567", "pwd": "123456", "role": "normal", "user_type": "passenger"},
    {"id": "pax_zhou",  "name": "周老师", "phone": "177****0123", "pwd": "123456", "role": "normal", "user_type": "passenger"},
]

# 预设行程起点→终点 (索引对应 JINAN_LANDMARKS)
TRIP_SEED_CONFIGS = [
    (0, 2),   # 泉城广场 → 大明湖
    (1, 3),   # 趵突泉 → 济南西站
    (4, 0),   # 遥墙机场 → 泉城广场
    (5, 6),   # 千佛山 → 高新区万达
    (3, 5),   # 济南西站 → 千佛山
]

def seed_initial_trips():
    """系统启动时预生成乘客账号与待接订单到服务器内存缓存"""
    with taxi_state_lock:
        # 清理过期订单 (超过30分钟未接)
        now = time.time()
        expired = [tid for tid, t in TAXI_TRIPS.items()
                   if t["status"] == "searching" and now - t.get("created_at", 0) > 1800]
        for tid in expired:
            del TAXI_TRIPS[tid]

        # 注册预置乘客账号到内存
        for pax in PASSENGER_SEED_DATA:
            SEED_PASSENGERS[pax["id"]] = pax

        # 已有足够搜索中订单则跳过
        existing = [t for t in TAXI_TRIPS.values() if t["status"] == "searching"]
        if len(existing) >= 5:
            return

        # 按时间戳生成唯一订单ID, 逐个写入
        base_ts = int(time.time())
        for i, (pax, (pu_idx, do_idx)) in enumerate(zip(PASSENGER_SEED_DATA, TRIP_SEED_CONFIGS)):
            pu = JINAN_LANDMARKS[pu_idx]
            do = JINAN_LANDMARKS[do_idx]
            dist = haversine_distance(pu["lng"], pu["lat"], do["lng"], do["lat"])
            price = round(max(10, dist * 2.8 + 8 + random.uniform(-3, 5)), 2)

            trip_id = f"seed_{base_ts}_{i}"
            TAXI_TRIPS[trip_id] = {
                "id": trip_id,
                "passenger_id": pax["id"],
                "passenger_name": pax["name"],
                "passenger_phone": pax["phone"],
                "pickup": {"name": pu["name"], "lng": pu["lng"], "lat": pu["lat"]},
                "dropoff": {"name": do["name"], "lng": do["lng"], "lat": do["lat"]},
                "pickup_lng": pu["lng"],
                "pickup_lat": pu["lat"],
                "dropoff_lng": do["lng"],
                "dropoff_lat": do["lat"],
                "status": "searching",
                "price": price,
                "distance": round(dist, 2),
                "duration": max(5, int(dist * 2)),
                "ride_type": "sd_xuan",
                "driver_id": None,
                "driver_name": None,
                "driver_location": None,
                "chat": [],
                "created_at": time.time()
            }

# ================= 2. 预测大盘数据集读取 =================
def get_current_dataset():
    return session.get('dataset', 'results')

def get_experiment_data(dataset_override=None):
    dataset_dir = dataset_override if dataset_override else get_current_dataset()
    json_path = os.path.join(CURRENT_DIR, dataset_dir, 'experiment_report.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"JSON 解析错误: {e}")
    return {}

# ================= 3. 系统核心路由 =================

@app.route('/')
def index(): return render_template('index.html', page="index")

@app.route('/login')
def login(): return render_template('login.html', page="login")

@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html', page="dashboard")

@app.route('/demand')
def demand(): return render_template('realtime1.html', page="demand")

@app.route('/dispatch')
def dispatch(): return render_template('dispatch.html', page="dispatch")

@app.route('/orders')
def orders(): return render_template('orders.html', page="orders")

# 司机接单界面 (替代原 taxi.html)
@app.route('/driver')
def driver():
    seed_ai_taxi_driver()
    seed_initial_trips()
    return render_template('driver.html', page="driver", landmarks=JINAN_LANDMARKS, vehicles=VEHICLE_TYPES)

# 乘客打车界面 (新)
@app.route('/passenger')
def passenger():
    seed_ai_taxi_driver()
    seed_initial_trips()
    return render_template('passenger.html', page="passenger", landmarks=JINAN_LANDMARKS, vehicles=VEHICLE_TYPES)

# 保留旧路由兼容
@app.route('/taxi')
def taxi():
    return render_template('driver.html', page="driver", landmarks=JINAN_LANDMARKS, vehicles=VEHICLE_TYPES)

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

# 动态静态分发
@app.route('/results/<path:filename>')
@app.route('/results2/<path:filename>')
def serve_results_file(filename):
    # URL-decode the path segment to handle spaces encoded as %20
    dataset_dir = urllib.parse.unquote(request.path.split('/')[1])
    if dataset_dir == 'real-time':
        dataset_dir = 'real-time results'
    referer = request.headers.get("Referer", "")
    if "/analytics" in referer or "/contrast" in referer:
        dataset_dir = "results2"
    results_dir = os.path.join(CURRENT_DIR, dataset_dir)
    return send_from_directory(results_dir, filename)

# ================= 4. Mamba-GNN 预测数据接口 =================
@app.route('/api/data')
def api_data():
    referer = request.headers.get("Referer", "")
    dataset_param = request.args.get("dataset")

    if "/analytics" in referer or "/contrast" in referer:
        data = get_experiment_data(dataset_override="results2")
    elif dataset_param in ["results", "results2", "real-time results"]:
        data = get_experiment_data(dataset_override=dataset_param)
    else:
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


# ================= 5. 打车模拟器配套 API =================

def haversine_distance(lng1, lat1, lng2, lat2):
    """计算两点间的球面距离 (km)"""
    R = 6371
    dLat = math.radians(lat2 - lat1)
    dLng = math.radians(lng2 - lng1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLng/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def get_grid_demand_prediction(grid_id):
    """获取指定网格的需求预测值"""
    try:
        json_path = os.path.join(CURRENT_DIR, 'real-time results', 'experiment_report.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            regions = data.get('top_predicted_regions_analysis', [])
            tick = int(time.time()) % 50
            for region in regions:
                if region.get('grid_id') == grid_id:
                    preds = region.get('time_series_data', {}).get('predicted_values', [])
                    if preds and len(preds) > tick:
                        return float(preds[tick])
                    return float(preds[0]) if preds else 50.0
    except:
        pass
    return 50.0

# 5.1 乘客发起叫车 (自动派单)
@app.route('/api/trips', methods=['POST'])
def create_trip():
    data = request.json
    trip_id = str(uuid.uuid4())[:8]

    pickup = data.get('pickup', {})
    dropoff = data.get('dropoff', {})
    pickup_lng = pickup.get('lng', 117.024967)
    pickup_lat = pickup.get('lat', 36.661156)
    dropoff_lng = dropoff.get('lng', 117.025354)
    dropoff_lat = dropoff.get('lat', 36.672957)

    distance = haversine_distance(pickup_lng, pickup_lat, dropoff_lng, dropoff_lat)
    price = data.get('price', round(max(8, distance * 2.8 + 6), 2))

    new_trip = {
        "id": trip_id,
        "passenger_id": data.get("passenger_id", "anonymous_passenger"),
        "passenger_name": data.get("passenger_name", "泉城旅客"),
        "pickup": pickup,
        "dropoff": dropoff,
        "pickup_lng": pickup_lng,
        "pickup_lat": pickup_lat,
        "dropoff_lng": dropoff_lng,
        "dropoff_lat": dropoff_lat,
        "status": "searching",
        "price": price,
        "distance": round(distance, 2),
        "duration": data.get("duration", max(5, int(distance * 2))),
        "ride_type": data.get("ride_type", "sd_xuan"),
        "driver_id": None,
        "driver_name": None,
        "driver_location": None,
        "chat": [],
        "created_at": time.time()
    }

    # 自动派单: 寻找最佳司机
    with taxi_state_lock:
        TAXI_TRIPS[trip_id] = new_trip

        # 筛选在线且空闲的司机
        online_drivers = {
            did: d for did, d in TAXI_DRIVERS.items()
            if d.get("status") == "online" and d.get("user_type") == "driver"
        }

        if online_drivers:
            # 获取起终点网格需求预测
            def get_grid_id(lng, lat):
                lon_min, lon_max = 116.0, 118.0
                lat_min, lat_max = 36.0, 37.8
                x = int((lng - lon_min) / ((lon_max - lon_min) / 15))
                y = int((lat - lat_min) / ((lat_max - lat_min) / 15))
                return max(0, min(14, y)) * 15 + max(0, min(14, x))

            pickup_grid = get_grid_id(pickup_lng, pickup_lat)
            dropoff_grid = get_grid_id(dropoff_lng, dropoff_lat)
            pickup_demand = get_grid_demand_prediction(pickup_grid)
            dropoff_demand = get_grid_demand_prediction(dropoff_grid)

            # 对每个在线司机打分
            best_driver = None
            best_score = -1
            for did, driver in online_drivers.items():
                drv_lng = driver["location"]["lng"]
                drv_lat = driver["location"]["lat"]
                dist_to_pickup = haversine_distance(drv_lng, drv_lat, pickup_lng, pickup_lat)

                # 综合评分: 距离近(50%) + 起点需求高(15%) + 终点需求高(15%) + 金额高(20%)
                dist_score = max(0, 100 - dist_to_pickup * 20)
                demand_pickup_score = min(100, pickup_demand * 0.5)
                demand_dropoff_score = min(100, dropoff_demand * 0.5)
                price_score = min(100, float(price) * 2)

                total_score = (dist_score * 0.5 + demand_pickup_score * 0.15 +
                               demand_dropoff_score * 0.15 + price_score * 0.2)

                if total_score > best_score:
                    best_score = total_score
                    best_driver = (did, driver)

            # 自动指派最佳司机
            if best_driver and best_score > 10:
                did, driver = best_driver
                new_trip["status"] = "accepted"
                new_trip["driver_id"] = did
                new_trip["driver_name"] = driver["name"]
                new_trip["driver_vehicle"] = driver.get("vehicle", "")
                new_trip["driver_rating"] = driver.get("rating", 0)
                new_trip["driver_location"] = driver["location"]
                TAXI_DRIVERS[did]["status"] = "busy"


    return jsonify({"success": True, "trip": new_trip})

# 5.2 获取行程详情
@app.route('/api/trips/<trip_id>', methods=['GET'])
def get_trip(trip_id):
    with taxi_state_lock:
        trip = TAXI_TRIPS.get(trip_id)
        if not trip:
            return jsonify({"success": False, "message": "打车行程未找到"}), 404
        return jsonify({"success": True, "trip": trip})

# 5.3 乘客获取自己的行程列表
@app.route('/api/passenger/trips', methods=['GET'])
def get_passenger_trips():
    passenger_id = request.args.get('passenger_id', '')
    with taxi_state_lock:
        if passenger_id:
            trips = [t for t in TAXI_TRIPS.values() if t.get("passenger_id") == passenger_id]
        else:
            trips = list(TAXI_TRIPS.values())
        return jsonify({"success": True, "trips": trips})

# 5.4 司机端: 获取智能排序的待接订单列表 (核心自动派单算法)
@app.route('/api/driver/available-trips', methods=['GET'])
def get_driver_available_trips():
    driver_lng = float(request.args.get('lng', 117.024967))
    driver_lat = float(request.args.get('lat', 36.661156))

    with taxi_state_lock:
        searching_trips = [t for t in TAXI_TRIPS.values() if t["status"] == "searching"]

    def _get_grid_id(lng, lat):
        lon_min, lon_max = 116.0, 118.0
        lat_min, lat_max = 36.0, 37.8
        x = int((lng - lon_min) / ((lon_max - lon_min) / 15))
        y = int((lat - lat_min) / ((lat_max - lat_min) / 15))
        return max(0, min(14, y)) * 15 + max(0, min(14, x))

    scored_trips = []
    for trip in searching_trips:
        pickup_lng = trip.get('pickup_lng', trip.get('pickup', {}).get('lng', 117.024967))
        pickup_lat = trip.get('pickup_lat', trip.get('pickup', {}).get('lat', 36.661156))
        dropoff_lng = trip.get('dropoff_lng', trip.get('dropoff', {}).get('lng', 117.025354))
        dropoff_lat = trip.get('dropoff_lat', trip.get('dropoff', {}).get('lat', 36.672957))

        dist_to_pickup = haversine_distance(driver_lng, driver_lat, pickup_lng, pickup_lat)
        price = float(trip.get('price', 15.0))

        pickup_grid = _get_grid_id(pickup_lng, pickup_lat)
        dropoff_grid = _get_grid_id(dropoff_lng, dropoff_lat)
        pickup_demand = get_grid_demand_prediction(pickup_grid)
        dropoff_demand = get_grid_demand_prediction(dropoff_grid)

        # 综合评分: 距离(50%) + 起点需求(15%) + 终点需求(15%) + 金额(20%)
        distance_score = max(0, 100 - dist_to_pickup * 20)
        demand_pickup_score = min(100, pickup_demand * 0.5)
        demand_dropoff_score = min(100, dropoff_demand * 0.5)
        price_score = min(100, price * 2)

        total_score = (distance_score * 0.5 + demand_pickup_score * 0.15 +
                       demand_dropoff_score * 0.15 + price_score * 0.2)

        scored_trips.append({
            **trip,
            "dist_to_driver": round(dist_to_pickup, 3),
            "pickup_demand": round(pickup_demand, 1),
            "dropoff_demand": round(dropoff_demand, 1),
            "auto_score": round(total_score, 1),
            "distance_score": round(distance_score, 1),
            "demand_pickup_score": round(demand_pickup_score, 1),
            "demand_dropoff_score": round(demand_dropoff_score, 1),
            "price_score": round(price_score, 1)
        })

    scored_trips.sort(key=lambda t: t['auto_score'], reverse=True)
    return jsonify({"success": True, "trips": scored_trips})

# 5.4b 司机端: 获取已指派给自己的订单
@app.route('/api/driver/assigned-trip', methods=['GET'])
def get_driver_assigned_trip():
    driver_id = request.args.get('driver_id', '')
    with taxi_state_lock:
        for trip in TAXI_TRIPS.values():
            if (trip.get("driver_id") == driver_id and
                    trip["status"] in ("accepted", "arrived", "driving")):
                return jsonify({"success": True, "trip": trip})
        return jsonify({"success": True, "trip": None})

# 5.5 司机抢单
@app.route('/api/trips/<trip_id>/accept', methods=['POST'])
def accept_trip(trip_id):
    data = request.json
    driver_id = data.get("driver_id")
    driver_name = data.get("driver_name", "鲁A神秘司机")
    driver_loc = data.get("driver_location")

    with taxi_state_lock:
        trip = TAXI_TRIPS.get(trip_id)
        if not trip:
            return jsonify({"success": False, "message": "该打车行程已失效"}), 404
        if trip["status"] != "searching":
            return jsonify({"success": False, "message": "该行程已被其他车主抢先承接"}), 400

        trip["status"] = "accepted"
        trip["driver_id"] = driver_id
        trip["driver_name"] = driver_name
        trip["driver_location"] = driver_loc

        if driver_id in TAXI_DRIVERS:
            TAXI_DRIVERS[driver_id]["status"] = "busy"
            trip["driver_vehicle"] = TAXI_DRIVERS[driver_id].get("vehicle", "")
            trip["driver_rating"] = TAXI_DRIVERS[driver_id].get("rating", 0)


        return jsonify({"success": True, "trip": trip})

# 5.6 司机更新行程状态
@app.route('/api/trips/<trip_id>/status', methods=['POST'])
def update_trip_status(trip_id):
    data = request.json
    new_status = data.get("status")
    driver_loc = data.get("driver_location")

    with taxi_state_lock:
        trip = TAXI_TRIPS.get(trip_id)
        if not trip:
            return jsonify({"success": False, "message": "未找到指定行程"}), 404

        trip["status"] = new_status
        if driver_loc:
            trip["driver_location"] = driver_loc

        if new_status == "completed" and trip["driver_id"] in TAXI_DRIVERS:
            TAXI_DRIVERS[trip["driver_id"]]["status"] = "online"
        if new_status == "completed":
            trip["paid"] = False  # 初始标记为未付款


        return jsonify({"success": True, "trip": trip})

# 5.7 乘客付款确认
@app.route('/api/trips/<trip_id>/pay', methods=['POST'])
def pay_trip(trip_id):
    with taxi_state_lock:
        trip = TAXI_TRIPS.get(trip_id)
        if not trip:
            return jsonify({"success": False, "message": "未找到指定行程"}), 404
        trip["paid"] = True
        trip["paid_at"] = time.time()

        return jsonify({"success": True, "trip": trip})

# 5.8 司机查询已完成的行程（含付款状态）
@app.route('/api/driver/trips', methods=['GET'])
def get_driver_trips():
    driver_id = request.args.get("driver_id", "")
    if not driver_id:
        return jsonify({"success": False, "trips": []})
    with taxi_state_lock:
        trips = [t for t in TAXI_TRIPS.values()
                 if t.get("driver_id") == driver_id]
    return jsonify({"success": True, "trips": trips})

@app.route('/api/admin/all-trips', methods=['GET'])
def get_all_trips_admin():
    """管理员/高级用户获取所有行程（含乘客订单和调度订单）"""
    with taxi_state_lock:
        all_trips = list(TAXI_TRIPS.values())
        # 附加调度订单信息
        for trip in all_trips:
            dispatch_id = trip.get("dispatch_order_id")
            if dispatch_id and dispatch_id in DISPATCH_ORDERS:
                trip["dispatch_info"] = {
                    "grid_id": DISPATCH_ORDERS[dispatch_id]["grid_id"],
                    "grid_name": DISPATCH_ORDERS[dispatch_id]["grid_name"],
                    "needed": DISPATCH_ORDERS[dispatch_id]["needed"],
                    "accepted": DISPATCH_ORDERS[dispatch_id]["accepted"]
                }
        all_trips.sort(key=lambda t: t.get("created_at", 0), reverse=True)
    return jsonify({"success": True, "trips": all_trips})

# 5.7 车载实时在线即时通信
@app.route('/api/trips/<trip_id>/chat', methods=['POST'])
def send_chat(trip_id):
    data = request.json
    sender_id = data.get("sender_id")
    sender_name = data.get("sender_name")
    text = data.get("text")

    with taxi_state_lock:
        trip = TAXI_TRIPS.get(trip_id)
        if not trip:
            return jsonify({"success": False, "message": "未找到行程"}), 404

        msg = {
            "sender_id": sender_id,
            "sender_name": sender_name,
            "text": text,
            "timestamp": time.strftime("%H:%M:%S")
        }
        trip["chat"].append(msg)
        return jsonify({"success": True, "chat": trip["chat"]})

# 5.8 司机端位置及可用状态心跳上报
@app.route('/api/drivers/heartbeat', methods=['POST'])
def driver_heartbeat():
    data = request.json
    driver_id = data.get("driver_id")
    name = data.get("name")
    location = data.get("location")
    status = data.get("status", "online")
    vehicle = data.get("vehicle", "鲁A·绿牌纯电轿车")
    user_type = data.get("user_type", "driver")

    with taxi_state_lock:
        # 保留已有名称和评分（种子司机有预配置的漂亮名称和评分）
        existing = TAXI_DRIVERS.get(driver_id)
        TAXI_DRIVERS[driver_id] = {
            "driver_id": driver_id,
            "name": existing["name"] if existing else name,
            "location": location,
            "status": status,
            "vehicle": existing["vehicle"] if existing else vehicle,
            "user_type": user_type,
            "rating": existing["rating"] if existing else 4.95,
            "updated_at": time.time()
        }
    return jsonify({"success": True})

# 5.9 乘客拉取地图上的在线空闲司机
@app.route('/api/drivers', methods=['GET'])
def get_online_drivers():
    with taxi_state_lock:
        now = time.time()
        active_drivers = [d for d in TAXI_DRIVERS.values()
                          if d.get("status") == "online" and now - d["updated_at"] < 30]
        return jsonify({"success": True, "drivers": active_drivers})

# 5.10 乘客端获取行程历史 (从 localStorage 同步到服务端)
@app.route('/api/passenger/history', methods=['GET'])
def get_passenger_history():
    passenger_id = request.args.get('passenger_id', '')
    with taxi_state_lock:
        all_trips = list(TAXI_TRIPS.values())
        if passenger_id:
            history = [t for t in all_trips if t.get("passenger_id") == passenger_id]
        else:
            history = all_trips
        return jsonify({"success": True, "history": history})


# ================= 6. 智能调度指令联动中继接口 =================

@app.route('/api/dispatch/alert', methods=['POST'])
def post_dispatch_alert():
    global LATEST_DISPATCH_ALERT
    data = request.json
    with taxi_state_lock:
        LATEST_DISPATCH_ALERT = {
            "grid_id": data.get("grid_id"),
            "grid_name": data.get("grid_name"),
            "count": data.get("count", 15),
            "target_time": data.get("target_time", "--:--"),
            "lon": data.get("lon"),
            "lat": data.get("lat"),
            "timestamp": data.get("timestamp", int(time.time() * 1000))
        }
    return jsonify({"success": True, "alert": LATEST_DISPATCH_ALERT})

@app.route('/api/dispatch/alert', methods=['GET'])
def get_dispatch_alert():
    with taxi_state_lock:
        if LATEST_DISPATCH_ALERT:
            return jsonify({"success": True, "alert": LATEST_DISPATCH_ALERT})
        return jsonify({"success": True, "alert": None})


# ================= 7. 网格调度广播接口（订单分发给所有司机，支持抢单+缺口补齐） =================

@app.route('/api/dispatch/broadcast', methods=['POST'])
def broadcast_dispatch_order():
    """调度页创建网格调度订单，广播给所有在线司机"""
    data = request.json
    grid_id = data.get("grid_id")
    grid_name = data.get("grid_name", f"网格 #{grid_id}")
    lng = data.get("lng", 117.024967)
    lat = data.get("lat", 36.661156)
    needed = max(1, int(data.get("needed", 1)))
    price = float(data.get("price", 30))

    order_id = f"ds_{int(time.time())}_{random.randint(1000, 9999)}"
    with taxi_state_lock:
        DISPATCH_ORDERS[order_id] = {
            "id": order_id,
            "grid_id": grid_id,
            "grid_name": grid_name,
            "lng": lng,
            "lat": lat,
            "needed": needed,
            "accepted": 0,
            "accepted_by": [],
            "price_estimate": round(price, 2),
            "status": "broadcasting",
            "created_at": time.time()
        }
    return jsonify({"success": True, "order": DISPATCH_ORDERS[order_id]})

@app.route('/api/dispatch/orders', methods=['GET'])
def get_dispatch_orders():
    """获取所有广播中的调度订单（司机端轮询，过滤已忽略的）"""
    driver_id = request.args.get('driver_id', '')
    with taxi_state_lock:
        orders = [
            {**o, "accepted_by": o["accepted_by"][:]}
            for o in DISPATCH_ORDERS.values()
            if o["status"] == "broadcasting"
        ]
        # 过滤掉该司机已忽略的订单
        if driver_id and driver_id in IGNORED_DISPATCH_ORDERS:
            ignored = IGNORED_DISPATCH_ORDERS[driver_id]
            orders = [o for o in orders if o["id"] not in ignored]
        orders.sort(key=lambda o: o["created_at"], reverse=True)
    return jsonify({"success": True, "orders": orders})

@app.route('/api/dispatch/orders/all', methods=['GET'])
def get_all_dispatch_orders():
    """获取所有调度订单（调度页轮询，含已完成/已取消）"""
    with taxi_state_lock:
        orders = [
            {**o, "accepted_by": o["accepted_by"][:]}
            for o in DISPATCH_ORDERS.values()
        ]
        orders.sort(key=lambda o: o["created_at"], reverse=True)
    return jsonify({"success": True, "orders": orders})

@app.route('/api/dispatch/orders/<order_id>/accept', methods=['POST'])
def accept_dispatch_order(order_id):
    """司机接受一条调度订单，同时创建对应行程"""
    data = request.json
    driver_id = data.get("driver_id")
    driver_name = data.get("driver_name", "司机")
    driver_loc = data.get("driver_location")

    with taxi_state_lock:
        order = DISPATCH_ORDERS.get(order_id)
        if not order:
            return jsonify({"success": False, "message": "订单不存在或已过期"}), 404
        if order["status"] != "broadcasting":
            return jsonify({"success": False, "message": "该订单已结束"}), 400

        # 检查是否已接过
        if any(a["driver_id"] == driver_id for a in order["accepted_by"]):
            return jsonify({"success": False, "message": "您已接过此单"}), 400

        # 检查司机是否已在其他调度订单中
        for o in DISPATCH_ORDERS.values():
            if o["id"] != order_id and o["status"] == "broadcasting":
                if any(a["driver_id"] == driver_id for a in o["accepted_by"]):
                    return jsonify({"success": False, "message": "您已有其他调度任务，请先完成"}), 400

        order["accepted_by"].append({
            "driver_id": driver_id,
            "driver_name": driver_name,
            "accepted_at": time.time()
        })
        order["accepted"] = len(order["accepted_by"])

        # 缺口补齐 → 自动结束广播
        if order["accepted"] >= order["needed"]:
            order["status"] = "filled"

        # 为接单司机创建实际行程
        trip_id = f"dispatch_{order_id}_{driver_id}"
        driver_info = TAXI_DRIVERS.get(driver_id, {})
        TAXI_TRIPS[trip_id] = {
            "id": trip_id,
            "passenger_id": "dispatch_sys",
            "passenger_name": f"调度任务 · {order['grid_name']}",
            "pickup": {"name": "当前位置", "lng": driver_loc["lng"] if driver_loc else order["lng"], "lat": driver_loc["lat"] if driver_loc else order["lat"]},
            "dropoff": {"name": order["grid_name"], "lng": order["lng"], "lat": order["lat"]},
            "pickup_lng": driver_loc["lng"] if driver_loc else order["lng"],
            "pickup_lat": driver_loc["lat"] if driver_loc else order["lat"],
            "dropoff_lng": order["lng"],
            "dropoff_lat": order["lat"],
            "status": "accepted",
            "price": 0,
            "distance": round(haversine_distance(
                driver_loc["lng"] if driver_loc else order["lng"],
                driver_loc["lat"] if driver_loc else order["lat"],
                order["lng"], order["lat"]
            ), 2),
            "duration": max(5, int(order.get("needed", 1) * 5)),
            "ride_type": "dispatch_broadcast",
            "driver_id": driver_id,
            "driver_name": driver_name,
            "driver_vehicle": driver_info.get("vehicle", ""),
            "driver_rating": driver_info.get("rating", 0),
            "driver_location": driver_loc,
            "dispatch_order_id": order_id,
            "chat": [],
            "created_at": time.time()
        }

        if driver_id in TAXI_DRIVERS:
            TAXI_DRIVERS[driver_id]["status"] = "busy"


        return jsonify({"success": True, "order": order, "trip": TAXI_TRIPS[trip_id]})

@app.route('/api/dispatch/orders/<order_id>/ignore', methods=['POST'])
def ignore_dispatch_order(order_id):
    """司机忽略一条调度订单"""
    data = request.json
    driver_id = data.get("driver_id")
    if not driver_id:
        return jsonify({"success": False, "message": "缺少司机ID"}), 400
    with taxi_state_lock:
        if driver_id not in IGNORED_DISPATCH_ORDERS:
            IGNORED_DISPATCH_ORDERS[driver_id] = set()
        IGNORED_DISPATCH_ORDERS[driver_id].add(order_id)
    return jsonify({"success": True})

@app.route('/api/dispatch/orders/<order_id>/cancel', methods=['POST'])
def cancel_dispatch_order(order_id):
    """取消一条调度订单（调度员操作）"""
    with taxi_state_lock:
        order = DISPATCH_ORDERS.get(order_id)
        if not order:
            return jsonify({"success": False, "message": "订单不存在"}), 404
        order["status"] = "cancelled"
        return jsonify({"success": True})

# 更新行程取消接口 — 支持乘客和司机双方取消
@app.route('/api/trips/<trip_id>/cancel', methods=['POST'])
def cancel_trip(trip_id):
    """乘客或司机取消行程"""
    data = request.json
    cancelled_by = data.get("cancelled_by", "passenger")  # passenger | driver

    with taxi_state_lock:
        trip = TAXI_TRIPS.get(trip_id)
        if not trip:
            return jsonify({"success": False, "message": "行程不存在"}), 404

        if trip["status"] in ("completed", "cancelled"):
            return jsonify({"success": False, "message": "行程已结束，无法取消"}), 400

        trip["status"] = "cancelled"
        trip["cancelled_by"] = cancelled_by

        # 释放司机
        driver_id = trip.get("driver_id")
        if driver_id and driver_id in TAXI_DRIVERS:
            TAXI_DRIVERS[driver_id]["status"] = "online"


        return jsonify({"success": True, "trip": trip})


if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)