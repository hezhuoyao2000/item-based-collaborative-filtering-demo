# -*- coding: utf-8 -*-
"""
Web API 接口
提供简单的HTTP接口用于前端展示
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import random
from data_manager import data_manager
from recommender import Recommender

app = Flask(__name__, static_folder='static')
CORS(app)

# 全局变量，避免重复加载
recommender_instance = None
processor_instance = None

def get_recommender():
    """获取推荐器实例（单例模式）"""
    global recommender_instance, processor_instance
    
    if recommender_instance is None:
        # 确保数据已加载
        processor_instance = data_manager.ensure_data_loaded()
        recommender_instance = Recommender()
    
    return recommender_instance, processor_instance

@app.route('/')
def index():
    """返回主页面"""
    return send_from_directory('static', 'index.html')

@app.route('/api/simulate_user', methods=['POST'])
def api_simulate_user():
    """模拟一个用户的交互行为"""
    try:
        # 获取数据处理器
        _, processor = get_recommender()
        
        # 随机选择一个用户
        all_users = processor.train_df['user_id'].unique()
        user_id = random.choice(all_users)
        
        # 获取用户交互历史（限制显示10条）
        user_interactions = processor.get_user_interactions(user_id, 'train')
        
        # 转换为列表格式
        interactions_list = []
        for _, row in user_interactions.head(10).iterrows():
            # 获取商品信息
            item_info = processor.items_df[processor.items_df['item_id'] == row['item_id']]
            category = item_info['category'].iloc[0] if len(item_info) > 0 else 'Unknown'
            
            interactions_list.append({
                'item_id': row['item_id'],
                'interaction_type': row['interaction_type'],
                'category': category,
                'timestamp': int(row['timestamp'])
            })
        
        # 获取用户统计信息
        total_interactions = len(user_interactions)
        interaction_types = user_interactions['interaction_type'].value_counts().to_dict()
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'total_interactions': total_interactions,
            'interaction_types': interaction_types,
            'interactions': interactions_list
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/recommend', methods=['GET'])
def api_recommend():
    """为指定用户生成推荐"""
    try:
        user_id = request.args.get('user_id')
        
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'user_id is required'
            }), 400
        
        # 获取推荐器
        recommender, processor = get_recommender()
        
        # 生成推荐
        recommendations = recommender.get_recommendations(user_id, top_n=10, data_processor=processor)
        
        if not recommendations:
            return jsonify({
                'success': False,
                'error': 'No recommendations available'
            }), 404
        
        # 获取推荐解释
        explanations = recommender.get_recommendation_explanations(
            user_id, 
            recommendations, 
            processor, 
            top_similar_items=3
        )
        
        # 格式化返回结果
        result = []
        for exp in explanations:
            result.append({
                'item_id': exp['recommended_item'],
                'score': round(exp['score'], 4),
                'category': exp.get('category', 'Unknown'),
                'parent_category': exp.get('parent_category', 'Unknown'),
                'price': round(exp['item_info'].get('price', 0), 2),
                'popularity': round(exp['item_info'].get('popularity_score', 0), 3),
                'explanation': exp['explanation'],
                'similar_items': exp['similar_items']
            })
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'recommendations': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def api_status():
    """检查系统状态"""
    try:
        data_ready = data_manager.check_data_files_exist()
        model_files = [
            os.path.join('models', 'similarity_matrix.npy'),
            os.path.join('models', 'item_index.pkl')
        ]
        model_ready = all(os.path.exists(f) for f in model_files)
        
        status_message = ''
        if not data_ready:
            status_message = '数据文件不存在，请先运行: python main.py generate_data'
        elif not model_ready:
            status_message = '模型文件不存在，请先运行: python main.py train_model'
        else:
            status_message = '系统就绪'
        
        return jsonify({
            'data_ready': data_ready,
            'model_ready': model_ready,
            'ready': data_ready and model_ready,
            'message': status_message
        })
    
    except Exception as e:
        return jsonify({
            'ready': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("="*60)
    print("启动推荐系统 Web 演示服务")
    print("="*60)
    
    # 检查系统状态
    if not data_manager.check_data_files_exist():
        print("⚠️  警告: 数据文件不存在")
        print("请先运行: python main.py generate_data")
        print()
    
    model_files = [
        os.path.join('models', 'similarity_matrix.npy'),
        os.path.join('models', 'item_index.pkl')
    ]
    if not all(os.path.exists(f) for f in model_files):
        print("⚠️  警告: 模型文件不存在")
        print("请先运行: python main.py train_model")
        print()
    
    print("✓ 服务启动成功!")
    print("📍 访问地址: http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, port=5000, host='0.0.0.0')

