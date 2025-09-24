# -*- coding: utf-8 -*-
"""
推荐系统主程序
Recommendation System Main Program
提供简洁的命令行接口，确保数据一致性
Provides a concise command-line interface to ensure data consistency
"""

import argparse
import sys
import os
from data_manager import data_manager
from model_trainer import ModelTrainer
from recommender import Recommender
import config

#region 数据生成模块 / Data Generation Module
def generate_data(force_regenerate=False):
    """生成模拟数据
    Generate simulated data
    
    Args:
        force_regenerate: 是否强制重新生成数据 / Whether to force data regeneration
    """
    print("="*60)
    print("数据生成模块 / Data Generation Module")
    print("="*60)
    
    # 检查数据文件是否已存在 / Check if data files already exist
    if data_manager.check_data_files_exist() and not force_regenerate:
        print("数据文件已存在，跳过生成步骤 / Data files already exist, skipping generation step")
        print("如需重新生成，请使用: python main.py generate_data --force / To regenerate, please use: python main.py generate_data --force")
        return None
    
    # 重置数据管理器确保重新生成 / Reset data manager to ensure regeneration
    data_manager.reset()
    processor = data_manager.get_processor()
    results = processor.generate_complete_dataset()
    data_manager.data_loaded = True
    data_manager.data_generated = True
    
    print("数据生成完成! / Data generation complete!")
    return results
#endregion

#region 模型训练模块 / Model Training Module

def train_model():
    """训练推荐模型 / Train the recommendation model"""
    print("="*60)
    print("模型训练模块 / Model Training Module")
    print("="*60)
    
    # 检查数据文件是否存在 / Check if data files exist
    if not data_manager.check_data_files_exist():
        print("错误: 数据文件不存在，请先运行: python main.py generate_data / Error: Data files do not exist, please run: python main.py generate_data first")
        return None
    
    # 确保数据已加载 / Ensure data is loaded
    processor = data_manager.ensure_data_loaded()
    
    # 检查训练数据是否有效 / Check if training data is valid
    if processor.user_item_matrix is None or processor.item_index is None:
        print("错误: 训练数据无效，请重新生成数据 / Error: Training data is invalid, please regenerate data")
        return None
    
    # 训练模型 / Train the model
    trainer = ModelTrainer()
    model_results = trainer.train(
        processor.user_item_matrix, 
        processor.item_index
    )
    
    print("模型训练完成! / Model training complete!")
    return model_results
#endregion

#region 模型评估模块 / Model Evaluation Module

def evaluate_model():
    """评估推荐模型 / Evaluate the recommendation model"""
    print("="*60)
    print("模型评估模块 / Model Evaluation Module")
    print("="*60)
    
    # 检查数据文件是否存在 / Check if data files exist
    if not data_manager.check_data_files_exist():
        print("错误: 数据文件不存在，请先运行: python main.py generate_data / Error: Data files do not exist, please run: python main.py generate_data first")
        return None
    
    # 检查模型是否存在 / Check if model exists
    model_files = [
        os.path.join(config.FILE_CONFIG['model_dir'], config.FILE_CONFIG['similarity_matrix_file']),
        os.path.join(config.FILE_CONFIG['model_dir'], config.FILE_CONFIG['item_index_file'])
    ]
    
    for model_file in model_files:
        if not os.path.exists(model_file):
            print(f"错误: 模型文件 {model_file} 不存在 / Error: Model file {model_file} does not exist")
            print("请先运行: python main.py train_model / Please run: python main.py train_model first")
            return None
    
    # 使用全局数据管理器 / Use global data manager
    processor = data_manager.ensure_data_loaded()
    
    # 检查测试数据是否有效 / Check if test data is valid
    if processor.test_df is None or len(processor.test_df) == 0:
        print("错误: 测试数据无效，请重新生成数据 / Error: Test data is invalid, please regenerate data")
        return None
    
    # 执行评估 / Perform evaluation
    recommender = Recommender()
    results = recommender.comprehensive_evaluate(max_users=100, data_processor=processor)
    
    if results:
        print("模型评估完成! / Model evaluation complete!")
    else:
        print("模型评估失败! / Model evaluation failed!")
    
    return results
#endregion

#region 用户推荐模块 / User Recommendation Module

def recommend_for_user(user_id, top_n=10):
    """为指定用户生成推荐 / Generate recommendations for a specified user"""
    print("="*60)
    print("用户推荐模块 / User Recommendation Module")
    print("="*60)
    
    # 检查数据文件是否存在 / Check if data files exist
    if not data_manager.check_data_files_exist():
        print("错误: 数据文件不存在，请先运行: python main.py generate_data / Error: Data files do not exist, please run: python main.py generate_data first")
        return None
    
    # 检查模型是否存在 / Check if model exists
    model_files = [
        os.path.join(config.FILE_CONFIG['model_dir'], config.FILE_CONFIG['similarity_matrix_file']),
        os.path.join(config.FILE_CONFIG['model_dir'], config.FILE_CONFIG['item_index_file'])
    ]
    
    for model_file in model_files:
        if not os.path.exists(model_file):
            print(f"错误: 模型文件 {model_file} 不存在 / Error: Model file {model_file} does not exist")
            print("请先运行: python main.py train_model / Please run: python main.py train_model first")
            return None
    
    try:
        # 创建推荐器 / Create recommender
        recommender = Recommender()
        
        # 使用全局数据管理器 / Use global data manager
        processor = data_manager.ensure_data_loaded()
        
        # 检查用户是否存在 / Check if user exists
        if user_id not in processor.interactions_df['user_id'].unique():
            print(f"错误: 用户 {user_id} 不存在 / Error: User {user_id} does not exist")
            return None
        
        print(f"为用户 {user_id} 生成推荐... / Generating recommendations for user {user_id}...")
        
        # 获取用户画像 / Get user profile
        profile = recommender.get_user_profile(user_id, processor)
        print(f"\n用户画像: / User Profile:")
        print(f"  用户ID: {profile['user_id']} /   User ID: {profile['user_id']}")
        print(f"  交互次数: {profile['interaction_count']} /   Interaction Count: {profile['interaction_count']}")
        print(f"  偏好类别: {profile['favorite_categories']} /   Favorite Categories: {profile['favorite_categories']}")
        print(f"  交互类型分布: {profile['interaction_types']} /   Interaction Type Distribution: {profile['interaction_types']}")
        
        # 生成推荐 / Generate recommendations
        recommendations = recommender.get_recommendations(user_id, top_n, processor)
        
        if recommendations:
            print(f"\n推荐结果 (Top-{top_n}): / Recommendations (Top-{top_n}):")
            for i, (item_id, score) in enumerate(recommendations, 1):
                print(f"  {i}. {item_id} (评分: {score:.4f}) /   {i}. {item_id} (Score: {score:.4f})")
            
            # 获取推荐解释 / Get recommendation explanations
            explanations = recommender.get_recommendation_explanations(user_id, recommendations, processor)
            print(f"\n推荐解释: / Recommendation Explanations:")
            for i, explanation in enumerate(explanations[:3], 1):
                print(f"  {i}. {explanation['explanation']} /   {i}. {explanation['explanation']}")
            
            # 显示推荐统计 / Display recommendation statistics
            stats = recommender.get_recommendation_stats(recommendations)
            print(f"\n推荐统计: / Recommendation Statistics:")
            print(f"  平均评分: {stats['avg_score']:.4f} /   Average Score: {stats['avg_score']:.4f}")
            print(f"  最高评分: {stats['max_score']:.4f} /   Maximum Score: {stats['max_score']:.4f}")
            print(f"  最低评分: {stats['min_score']:.4f} /   Minimum Score: {stats['min_score']:.4f}")
            
        else:
            print(f"无法为用户 {user_id} 生成推荐 / Unable to generate recommendations for user {user_id}")
            print("可能原因: / Possible reasons:")
            print("  1. 用户ID不存在 /   1. User ID does not exist")
            print("  2. 用户没有历史交互记录 /   2. User has no historical interaction records")
            print("  3. 用户交互的商品数量过少 /   3. User has too few interacted items")
        
        return recommendations
        
    except Exception as e:
        print(f"推荐生成失败: {e} / Recommendation generation failed: {e}")
        return None
#endregion

#region 系统状态检查模块 / System Status Check Module

def show_system_status():
    """显示系统状态 / Show system status"""
    print("="*60)
    print("系统状态检查 / System Status Check")
    print("="*60)
    
    # 检查数据文件 / Check data files
    data_files = [
        os.path.join(config.FILE_CONFIG['data_dir'], 'interactions.csv'),
        os.path.join(config.FILE_CONFIG['data_dir'], 'items.csv'),
        os.path.join(config.FILE_CONFIG['data_dir'], 'train_interactions.csv'),
        os.path.join(config.FILE_CONFIG['data_dir'], 'test_interactions.csv')
    ]
    
    print("数据文件状态: / Data File Status:")
    for data_file in data_files:
        if os.path.exists(data_file):
            size = os.path.getsize(data_file) / 1024  # KB
            print(f"  ✓ {data_file} ({size:.1f} KB) /   ✓ {data_file} ({size:.1f} KB)")
        else:
            print(f"  ✗ {data_file} (不存在) /   ✗ {data_file} (Does not exist)")
    
    # 检查模型文件 / Check model files
    model_files = [
        os.path.join(config.FILE_CONFIG['model_dir'], config.FILE_CONFIG['similarity_matrix_file']),
        os.path.join(config.FILE_CONFIG['model_dir'], config.FILE_CONFIG['item_index_file'])
    ]
    
    print("\n模型文件状态: / Model File Status:")
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / 1024  # KB
            print(f"  ✓ {model_file} ({size:.1f} KB) /   ✓ {model_file} ({size:.1f} KB)")
        else:
            print(f"  ✗ {model_file} (不存在) /   ✗ {model_file} (Does not exist)")
    
    # 检查配置 / Check configuration
    print("\n配置信息: / Configuration Information:")
    print(f"  用户数量: {config.DATA_CONFIG['num_users']} /   Number of users: {config.DATA_CONFIG['num_users']}")
    print(f"  商品数量: {config.DATA_CONFIG['num_items']} /   Number of items: {config.DATA_CONFIG['num_items']}")
    print(f"  交互数量: {config.DATA_CONFIG['num_interactions']} /   Number of interactions: {config.DATA_CONFIG['num_interactions']}")
    print(f"  训练集比例: {config.DATA_CONFIG['train_ratio']} /   Training set ratio: {config.DATA_CONFIG['train_ratio']}")
    print(f"  推荐数量: {config.MODEL_CONFIG['top_n']} /   Number of recommendations: {config.MODEL_CONFIG['top_n']}")
#endregion

#region 验证工作流程步骤模块 / Validate Workflow Step Module

def validate_workflow_step(step_name, required_files=None, required_data=None):
    """验证工作流程步骤的前置条件 / Validate preconditions for a workflow step"""
    print(f"验证 {step_name} 前置条件... / Validating {step_name} preconditions...")
    
    # 检查必需文件 / Check required files
    if required_files:
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"错误: 必需文件 {file_path} 不存在 / Error: Required file {file_path} does not exist")
                return False
    
    # 检查必需数据 / Check required data
    if required_data:
        for data_name, data_value in required_data.items():
            if data_value is None:
                print(f"错误: 必需数据 {data_name} 未加载 / Error: Required data {data_name} not loaded")
                return False
    
    print(f"✓ {step_name} 前置条件验证通过 / ✓ {step_name} preconditions validated")
    return True
#endregion

#region 运行完整工作流程模块 / Run Complete Workflow Module

def run_complete_workflow():
    """运行完整的工作流程 / Run the complete workflow"""
    print("="*60)
    print("运行完整推荐系统工作流程 / Running complete recommendation system workflow")
    print("="*60)
    
    try:
        # 1. 生成数据 / 1. Generate data
        print("\n步骤1: 生成数据 / Step 1: Generating data")
        data_result = generate_data()
        if not data_result:
            print("数据生成失败，终止工作流程 / Data generation failed, terminating workflow")
            return False
        
        # 验证数据生成结果 / Validate data generation results
        if not validate_workflow_step("数据生成", required_data={'interactions_df': data_result.get('interactions_df')}):
            return False
        
        # 2. 训练模型 / 2. Train model
        print("\n步骤2: 训练模型 / Step 2: Training model")
        model_result = train_model()
        if not model_result:
            print("模型训练失败，终止工作流程 / Model training failed, terminating workflow")
            return False
        
        # 验证模型训练结果 / Validate model training results
        model_files = [
            os.path.join(config.FILE_CONFIG['model_dir'], config.FILE_CONFIG['similarity_matrix_file']),
            os.path.join(config.FILE_CONFIG['model_dir'], config.FILE_CONFIG['item_index_file'])
        ]
        if not validate_workflow_step("模型训练", required_files=model_files):
            return False
        
        # 3. 评估模型 / 3. Evaluate model
        print("\n步骤3: 评估模型 / Step 3: Evaluating model")
        eval_result = evaluate_model()
        if not eval_result:
            print("模型评估失败，但继续执行后续步骤 / Model evaluation failed, but continuing with subsequent steps")
        
        # 4. 示例推荐 / 4. Example recommendations
        print("\n步骤4: 示例推荐 / Step 4: Example Recommendations")
        processor = data_manager.ensure_data_loaded()
        test_users = processor.interactions_df['user_id'].unique()[:3]
        
        for user_id in test_users:
            print(f"\n--- 为用户 {user_id} 生成推荐 --- / --- Generating recommendations for user {user_id} ---")
            recommend_for_user(user_id, top_n=5)
        
        print("\n" + "="*60)
        print("完整工作流程执行完成! / Complete workflow execution finished!")
        print("="*60)
        
        # 显示数据摘要 / Display data summary
        summary = data_manager.get_data_summary()
        print(f"\n数据摘要: / Data Summary:")
        print(f"  用户数: {summary['users']} /   Number of users: {summary['users']}")
        print(f"  商品数: {summary['items']} /   Number of items: {summary['items']}")
        print(f"  交互数: {summary['interactions']} /   Number of interactions: {summary['interactions']}")
        print(f"  训练集交互数: {summary['train_interactions']} /   Training set interactions: {summary['train_interactions']}")
        print(f"  测试集交互数: {summary['test_interactions']} /   Test set interactions: {summary['test_interactions']}")
        
    except Exception as e:
        print(f"工作流程执行失败: {e} / Workflow execution failed: {e}")
        return False
    
    return True
#endregion

#region 主函数模块 / Main Function Module

def main():
    """主函数 / Main function"""
    parser = argparse.ArgumentParser(
        description="推荐系统 - 基于物品协同过滤 / Recommendation System - Item-based Collaborative Filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
Usage Examples:
  python main.py run                    # 运行完整工作流程 / Run complete workflow
  python main.py generate_data          # 生成模拟数据（如果已存在则跳过） / Generate simulated data (skips if already exists)
  python main.py generate_data --force  # 强制重新生成数据 / Force regenerate data
  python main.py train_model            # 训练推荐模型（基于现有数据） / Train recommendation model (based on existing data)
  python main.py evaluate               # 评估模型效果（基于现有数据） / Evaluate model performance (based on existing data)
  python main.py recommend --user_id u123  # 为用户生成推荐（基于现有数据） / Generate recommendations for user (based on existing data)
  python main.py status                 # 显示系统状态 / Show system status

独立使用流程:
Independent Usage Flow:
  1. 首次运行: python main.py generate_data / 1. First run: python main.py generate_data
  2. 训练模型: python main.py train_model / 2. Train model: python main.py train_model
  3. 评估模型: python main.py evaluate / 3. Evaluate model: python main.py evaluate
  4. 生成推荐: python main.py recommend --user_id u123 / 4. Generate recommendations: python main.py recommend --user_id u123
  
  注意: 数据生成后，其他功能可以独立运行，无需重新生成数据
  Note: After data generation, other functions can run independently without regenerating data
        """
    )
    
    parser.add_argument(
        'command',
        choices=['run', 'generate_data', 'train_model', 'evaluate', 'recommend', 'status'],
        help='要执行的命令 / Command to execute'
    )
    
    parser.add_argument(
        '--user_id',
        type=str,
        help='用户ID (仅在recommend命令中使用) / User ID (only used with recommend command)'
    )
    
    parser.add_argument(
        '--top_n',
        type=int,
        default=10,
        help='推荐商品数量 (默认: 10) / Number of recommended items (default: 10)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新生成数据 (仅在generate_data命令中使用) / Force regenerate data (only used with generate_data command)'
    )
    
    args = parser.parse_args()
    
    # 执行对应命令 / Execute corresponding command
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
            print("错误: recommend命令需要指定 --user_id 参数 / Error: recommend command requires --user_id argument")
            print("示例: python main.py recommend --user_id u123 / Example: python main.py recommend --user_id u123")
            sys.exit(1)
        
        recommend_for_user(args.user_id, args.top_n)
    
    elif args.command == 'status':
        show_system_status()
    
    else:
        print(f"未知命令: {args.command} / Unknown command: {args.command}")
        sys.exit(1)
#endregion


if __name__ == "__main__":
    main()
