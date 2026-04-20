# -*- coding: utf-8 -*-
import os
import json
from flask import Flask, render_template, jsonify, send_from_directory, request, session

app = Flask(__name__)
app.secret_key = "mamba_gnn_super_secret_key"  # 必须设置 secret_key 才能使用 session
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_current_dataset():
    """获取当前用户选择的数据集，默认使用 7月1日-7月7日 (results)"""
    return session.get('dataset', 'results')

def get_experiment_data():
    """读取当前选择的数据集目录下的实验报告 JSON 数据"""
    dataset_dir = get_current_dataset()
    json_path = os.path.join(CURRENT_DIR, dataset_dir, 'experiment_report.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"JSON 解析错误: {e}")
    else:
        print(f"[警告] 找不到数据文件: {json_path}")
    return {}

# ================= 页面路由 =================
@app.route('/')
def index(): return render_template('index.html', page="index")

@app.route('/login')
def login(): return render_template('login.html', page="login")

@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html', page="dashboard")

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

# ================= 提供结果目录静态文件的路由 =================
@app.route('/results/<path:filename>')
def serve_results_file(filename):
    """
    允许前端直接访问当前 results 目录下的图片和文件。
    不管前端请求的是 /results/... 路径，我们都会将其动态映射到用户当前选择的文件夹 (results 或 results2)
    """
    dataset_dir = get_current_dataset()
    results_dir = os.path.join(CURRENT_DIR, dataset_dir)
    return send_from_directory(results_dir, filename)

# ================= API 路由 =================
@app.route('/api/data')
def api_data():
    """提供图表和面板所需的数据源"""
    data = get_experiment_data()
    if not data:
        return jsonify({"error": "Data file not found or empty"}), 404
    return jsonify(data)

@app.route('/api/set_dataset', methods=['POST'])
def set_dataset():
    """接收前端传来的时间范围切换请求"""
    data = request.get_json()
    if data and 'dataset' in data:
        dataset = data['dataset']
        if dataset in ['results', 'results2']:
            session['dataset'] = dataset
            return jsonify({"status": "success", "dataset": dataset})
    return jsonify({"error": "Invalid dataset"}), 400

@app.route('/api/get_dataset')
def api_get_dataset():
    """返回前端当前正在使用的数据集"""
    return jsonify({"dataset": get_current_dataset()})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
'''
    82: { "en": "Baotu Spring & City Center","cn": "趵突泉景区及市中心",
"url": "https://baike.baidu.com/item/%E8%B6%B5%E7%AA%81%E6%B3%89/162170"
    },  # 117.016158,36.660958
    83: {"en": "Shandong Museum & CBD","cn": "山东省博物馆及CBD区域",
        "url": "https://baike.baidu.com/item/%E5%B1%B1%E4%B8%9C%E5%8D%9A%E7%89%A9%E9%A6%86/9588493"
    },  # 117.095645,36.658497
    81: {"en": "Jixi National Wetland Park","cn": "山东济西国家湿地公园",
        "url": "https://baike.baidu.com/item/%E5%B1%B1%E4%B8%9C%E6%B5%8E%E8%A5%BF%E5%9B%BD%E5%AE%B6%E6%B9%BF%E5%9C%B0%E5%85%AC%E5%9B%AD/23591125"
    },  # 116.80664,36.652445
    86: {"en": "Baimai Spring Park","cn": "百脉泉公园",
        "url": "https://baike.baidu.com/item/%E7%99%BE%E8%84%89%E6%B3%89"
    } , # 117.53632,36.719829
    84: {"en": "Jinan CAPITAL OUTLETS","cn": "济南首创奥特莱斯",
        "url": "https://baike.baidu.com/item/%E9%A6%96%E5%88%9B%E5%A5%A5%E7%89%B9%E8%8E%B1%E6%96%AF/61224771"
    }  # 117.232234,36.692267
}
'''