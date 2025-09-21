# -*- coding: utf-8 -*-
"""
推荐系统主程序
提供简洁的命令行接口，确保数据一致性
"""

import argparse
import sys
import os
from data_manager import data_manager
from model_trainer import ModelTrainer
from recommender import Recommender
import config

#region 数据生成模块
def generate_data(force_regenerate=False):
    """生成模拟数据
    
    Args:
        force_regenerate: 是否强制重新生成数据
    """
    print("="*60)
    print("数据生成模块")
    print("="*60)
    
    # 检查数据文件是否已存在
    if data_manager.check_data_files_exist() and not force_regenerate:
        print("数据文件已存在，跳过生成步骤")
        print("如需重新生成，请使用: python main.py generate_data --force")
        return None
    
    # 重置数据管理器确保重新生成
    data_manager.reset()
    processor = data_manager.get_processor()
    results = processor.generate_complete_dataset()
    data_manager.data_loaded = True
    data_manager.data_generated = True
    
    print("数据生成完成!")
    return results
#endregion

#region 模型训练模块

def train_model():
    """训练推荐模型"""
    print("="*60)
    print("模型训练模块")
    print("="*60)
    
    # 检查数据文件是否存在
    if not data_manager.check_data_files_exist():
        print("错误: 数据文件不存在，请先运行: python main.py generate_data")
        return None
    
    # 确保数据已加载
    processor = data_manager.ensure_data_loaded()
    
    # 检查训练数据是否有效
    if processor.user_item_matrix is None or processor.item_index is None:
        print("错误: 训练数据无效，请重新生成数据")
        return None
    
    # 训练模型
    trainer = ModelTrainer()
    model_results = trainer.train(
        processor.user_item_matrix, 
        processor.item_index
    )
    
    print("模型训练完成!")
    return model_results
#endregion

#region 模型评估模块

def evaluate_model():
    """评估推荐模型"""
    print("="*60)
    print("模型评估模块")
    print("="*60)
    
    # 检查数据文件是否存在
    if not data_manager.check_data_files_exist():
        print("错误: 数据文件不存在，请先运行: python main.py generate_data")
        return None
    
    # 检查模型是否存在
    model_files = [
        os.path.join(config.FILE_CONFIG['model_dir'], config.FILE_CONFIG['similarity_matrix_file']),
        os.path.join(config.FILE_CONFIG['model_dir'], config.FILE_CONFIG['item_index_file'])
    ]
    
    for model_file in model_files:
        if not os.path.exists(model_file):
            print(f"错误: 模型文件 {model_file} 不存在")
            print("请先运行: python main.py train_model")
            return None
    
    # 使用全局数据管理器
    processor = data_manager.ensure_data_loaded()
    
    # 检查测试数据是否有效
    if processor.test_df is None or len(processor.test_df) == 0:
        print("错误: 测试数据无效，请重新生成数据")
        return None
    
    # 执行评估
    recommender = Recommender()
    results = recommender.comprehensive_evaluate(max_users=100, data_processor=processor)
    
    if results:
        print("模型评估完成!")
    else:
        print("模型评估失败!")
    
    return results
#endregion

#region 用户推荐模块

def recommend_for_user(user_id, top_n=10):
    """为指定用户生成推荐"""
    print("="*60)
    print("用户推荐模块")
    print("="*60)
    
    # 检查数据文件是否存在
    if not data_manager.check_data_files_exist():
        print("错误: 数据文件不存在，请先运行: python main.py generate_data")
        return None
    
    # 检查模型是否存在
    model_files = [
        os.path.join(config.FILE_CONFIG['model_dir'], config.FILE_CONFIG['similarity_matrix_file']),
        os.path.join(config.FILE_CONFIG['model_dir'], config.FILE_CONFIG['item_index_file'])
    ]
    
    for model_file in model_files:
        if not os.path.exists(model_file):
            print(f"错误: 模型文件 {model_file} 不存在")
            print("请先运行: python main.py train_model")
            return None
    
    try:
        # 创建推荐器
        recommender = Recommender()
        
        # 使用全局数据管理器
        processor = data_manager.ensure_data_loaded()
        
        # 检查用户是否存在
        if user_id not in processor.interactions_df['user_id'].unique():
            print(f"错误: 用户 {user_id} 不存在")
            return None
        
        print(f"为用户 {user_id} 生成推荐...")
        
        # 获取用户画像
        profile = recommender.get_user_profile(user_id, processor)
        print(f"\n用户画像:")
        print(f"  用户ID: {profile['user_id']}")
        print(f"  交互次数: {profile['interaction_count']}")
        print(f"  偏好类别: {profile['favorite_categories']}")
        print(f"  交互类型分布: {profile['interaction_types']}")
        
        # 生成推荐
        recommendations = recommender.get_recommendations(user_id, top_n, processor)
        
        if recommendations:
            print(f"\n推荐结果 (Top-{top_n}):")
            for i, (item_id, score) in enumerate(recommendations, 1):
                print(f"  {i}. {item_id} (评分: {score:.4f})")
            
            # 获取推荐解释
            explanations = recommender.get_recommendation_explanations(user_id, recommendations, processor)
            print(f"\n推荐解释:")
            for i, explanation in enumerate(explanations[:3], 1):
                print(f"  {i}. {explanation['explanation']}")
            
            # 显示推荐统计
            stats = recommender.get_recommendation_stats(recommendations)
            print(f"\n推荐统计:")
            print(f"  平均评分: {stats['avg_score']:.4f}")
            print(f"  最高评分: {stats['max_score']:.4f}")
            print(f"  最低评分: {stats['min_score']:.4f}")
            
        else:
            print(f"无法为用户 {user_id} 生成推荐")
            print("可能原因:")
            print("  1. 用户ID不存在")
            print("  2. 用户没有历史交互记录")
            print("  3. 用户交互的商品数量过少")
        
        return recommendations
        
    except Exception as e:
        print(f"推荐生成失败: {e}")
        return None
#endregion

#region 系统状态检查模块

def show_system_status():
    """显示系统状态"""
    print("="*60)
    print("系统状态检查")
    print("="*60)
    
    # 检查数据文件
    data_files = [
        os.path.join(config.FILE_CONFIG['data_dir'], 'enhanced_interactions.csv'),
        os.path.join(config.FILE_CONFIG['data_dir'], 'enhanced_items.csv'),
        os.path.join(config.FILE_CONFIG['data_dir'], 'train_interactions.csv'),
        os.path.join(config.FILE_CONFIG['data_dir'], 'test_interactions.csv')
    ]
    
    print("数据文件状态:")
    for data_file in data_files:
        if os.path.exists(data_file):
            size = os.path.getsize(data_file) / 1024  # KB
            print(f"  ✓ {data_file} ({size:.1f} KB)")
        else:
            print(f"  ✗ {data_file} (不存在)")
    
    # 检查模型文件
    model_files = [
        os.path.join(config.FILE_CONFIG['model_dir'], config.FILE_CONFIG['similarity_matrix_file']),
        os.path.join(config.FILE_CONFIG['model_dir'], config.FILE_CONFIG['item_index_file'])
    ]
    
    print("\n模型文件状态:")
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / 1024  # KB
            print(f"  ✓ {model_file} ({size:.1f} KB)")
        else:
            print(f"  ✗ {model_file} (不存在)")
    
    # 检查配置
    print("\n配置信息:")
    print(f"  用户数量: {config.DATA_CONFIG['num_users']}")
    print(f"  商品数量: {config.DATA_CONFIG['num_items']}")
    print(f"  交互数量: {config.DATA_CONFIG['num_interactions']}")
    print(f"  训练集比例: {config.DATA_CONFIG['train_ratio']}")
    print(f"  推荐数量: {config.MODEL_CONFIG['top_n']}")
#endregion

#region 验证工作流程步骤模块

def validate_workflow_step(step_name, required_files=None, required_data=None):
    """验证工作流程步骤的前置条件"""
    print(f"验证 {step_name} 前置条件...")
    
    # 检查必需文件
    if required_files:
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"错误: 必需文件 {file_path} 不存在")
                return False
    
    # 检查必需数据
    if required_data:
        for data_name, data_value in required_data.items():
            if data_value is None:
                print(f"错误: 必需数据 {data_name} 未加载")
                return False
    
    print(f"✓ {step_name} 前置条件验证通过")
    return True
#endregion

#region 运行完整工作流程模块

def run_complete_workflow():
    """运行完整的工作流程"""
    print("="*60)
    print("运行完整推荐系统工作流程")
    print("="*60)
    
    try:
        # 1. 生成数据
        print("\n步骤1: 生成数据")
        data_result = generate_data()
        if not data_result:
            print("数据生成失败，终止工作流程")
            return False
        
        # 验证数据生成结果
        if not validate_workflow_step("数据生成", required_data={'interactions_df': data_result.get('interactions_df')}):
            return False
        
        # 2. 训练模型
        print("\n步骤2: 训练模型")
        model_result = train_model()
        if not model_result:
            print("模型训练失败，终止工作流程")
            return False
        
        # 验证模型训练结果
        model_files = [
            os.path.join(config.FILE_CONFIG['model_dir'], config.FILE_CONFIG['similarity_matrix_file']),
            os.path.join(config.FILE_CONFIG['model_dir'], config.FILE_CONFIG['item_index_file'])
        ]
        if not validate_workflow_step("模型训练", required_files=model_files):
            return False
        
        # 3. 评估模型
        print("\n步骤3: 评估模型")
        eval_result = evaluate_model()
        if not eval_result:
            print("模型评估失败，但继续执行后续步骤")
        
        # 4. 示例推荐
        print("\n步骤4: 示例推荐")
        processor = data_manager.ensure_data_loaded()
        test_users = processor.interactions_df['user_id'].unique()[:3]
        
        for user_id in test_users:
            print(f"\n--- 为用户 {user_id} 生成推荐 ---")
            recommend_for_user(user_id, top_n=5)
        
        print("\n" + "="*60)
        print("完整工作流程执行完成!")
        print("="*60)
        
        # 显示数据摘要
        summary = data_manager.get_data_summary()
        print(f"\n数据摘要:")
        print(f"  用户数: {summary['users']}")
        print(f"  商品数: {summary['items']}")
        print(f"  交互数: {summary['interactions']}")
        print(f"  训练集交互数: {summary['train_interactions']}")
        print(f"  测试集交互数: {summary['test_interactions']}")
        
    except Exception as e:
        print(f"工作流程执行失败: {e}")
        return False
    
    return True
#endregion

#region 主函数模块

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="推荐系统 - 基于物品协同过滤",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py run                    # 运行完整工作流程
  python main.py generate_data          # 生成模拟数据（如果已存在则跳过）
  python main.py generate_data --force  # 强制重新生成数据
  python main.py train_model            # 训练推荐模型（基于现有数据）
  python main.py evaluate               # 评估模型效果（基于现有数据）
  python main.py recommend --user_id u123  # 为用户生成推荐（基于现有数据）
  python main.py status                 # 显示系统状态

独立使用流程:
  1. 首次运行: python main.py generate_data
  2. 训练模型: python main.py train_model
  3. 评估模型: python main.py evaluate
  4. 生成推荐: python main.py recommend --user_id u123
  
  注意: 数据生成后，其他功能可以独立运行，无需重新生成数据
        """
    )
    
    parser.add_argument(
        'command',
        choices=['run', 'generate_data', 'train_model', 'evaluate', 'recommend', 'status'],
        help='要执行的命令'
    )
    
    parser.add_argument(
        '--user_id',
        type=str,
        help='用户ID (仅在recommend命令中使用)'
    )
    
    parser.add_argument(
        '--top_n',
        type=int,
        default=10,
        help='推荐商品数量 (默认: 10)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新生成数据 (仅在generate_data命令中使用)'
    )
    
    args = parser.parse_args()
    
    # 执行对应命令
    if args.command == 'run':
        run_complete_workflow()
    
    elif args.command == 'generate_data':
        generate_data(force_regenerate=args.force)
    
    elif args.command == 'train_model':
        train_model()
    
    elif args.command == 'evaluate':
        evaluate_model()
    
    elif args.command == 'recommend':
        if not args.user_id:
            print("错误: recommend命令需要指定 --user_id 参数")
            print("示例: python main.py recommend --user_id u123")
            sys.exit(1)
        
        recommend_for_user(args.user_id, args.top_n)
    
    elif args.command == 'status':
        show_system_status()
    
    else:
        print(f"未知命令: {args.command}")
        sys.exit(1)
#endregion


if __name__ == "__main__":
    main()
