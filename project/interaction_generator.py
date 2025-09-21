# -*- coding: utf-8 -*-
"""
交互生成模块
基于用户兴趣生成用户-商品交互数据
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple
from config import DATA_CONFIG, RANDOM_SEED
from user_generator import UserGenerator
from item_generator import ItemGenerator

class InteractionGenerator:
    """交互生成器"""
    
    def __init__(self):
        """初始化交互生成器"""
        self.config = DATA_CONFIG
        self.num_interactions = self.config['num_interactions']
        self.interaction_types = self.config['interaction_types']
        self.interaction_weights = self.config['interaction_weights']
        
        # 设置随机种子
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        
        # 存储生成的交互数据
        self.interactions_df = None
        self.user_item_interactions = {}  # 用户-商品交互记录
    
    #region 生成交互类型  / region Generate interaction type
    def generate_interaction_type(self, user_id: str, item_category: str, 
                                user_generator: UserGenerator) -> str:
        """
        基于用户兴趣生成交互类型
        
        Args:
            user_id: 用户ID
            item_category: 商品分类
            user_generator: 用户生成器实例
            
        Returns:
            交互类型 ('click', 'cart', 'favorite')
        """
        # 获取用户对该分类的交互概率
        base_probability = user_generator.get_user_interaction_probability(user_id, item_category)
        
        # 基于基础概率决定是否生成交互
        if random.random() > base_probability:
            return None  # 不生成交互
        
        # 基于交互类型分布生成交互类型（所有用户都有机会生成不同类型）
        random_value = random.random()
        
        if random_value < self.interaction_types['click']:
            return 'click'
        elif random_value < self.interaction_types['click'] + self.interaction_types['cart']:
            return 'cart'
        else:
            return 'favorite'
    #endregion
    #region 为指定用户生成交互数据  / region Generate user interactions
    def generate_user_interactions(self, user_id: str, user_generator: UserGenerator, 
                                 item_generator: ItemGenerator) -> List[Dict]:
        """
        为指定用户生成交互数据
        
        Args:
            user_id: 用户ID
            user_generator: 用户生成器实例
            item_generator: 商品生成器实例
            
        Returns:
            用户交互数据列表
        """
        user_interactions = []
        
        # 获取用户偏好的分类
        preferred_categories = user_generator.get_user_preferred_categories(user_id)
        high_interest_categories = user_generator.get_user_high_interest_categories(user_id)
        
        # 计算用户交互数量范围
        min_interactions, max_interactions = self.config['interactions_per_user_range']
        num_interactions = random.randint(min_interactions, max_interactions)
        
        # 为每个交互选择商品
        for _ in range(num_interactions):
            # 选择分类（偏好分类有更高概率被选择）
            if preferred_categories and random.random() < 0.8:
                # 80%概率选择偏好分类
                if high_interest_categories and random.random() < 0.6:
                    # 60%概率选择高兴趣分类
                    category = random.choice(high_interest_categories)
                else:
                    # 40%概率选择其他偏好分类
                    category = random.choice(preferred_categories)
            else:
                # 20%概率随机选择分类（探索行为）
                all_categories = list(item_generator.category_items.keys())
                category = random.choice(all_categories)
            
            # 从分类中选择商品
            category_items = item_generator.get_items_by_category(category)
            if not category_items:
                continue
            
            item_id = random.choice(category_items)
            
            # 避免重复交互
            interaction_key = (user_id, item_id)
            if interaction_key in self.user_item_interactions:
                continue
            
            # 生成交互类型
            interaction_type = self.generate_interaction_type(user_id, category, user_generator)
            
            # 如果没有生成交互，跳过
            if interaction_type is None:
                continue
            
            # 记录交互
            interaction = {
                'user_id': user_id,
                'item_id': item_id,
                'interaction_type': interaction_type,
                'category': category
            }
            user_interactions.append(interaction)
            self.user_item_interactions[interaction_key] = interaction_type
        
        return user_interactions
    #endregion
    #region 生成所有用户的交互数据  / region Generate all interactions
    def generate_all_interactions(self, user_generator: UserGenerator, 
                                item_generator: ItemGenerator) -> pd.DataFrame:
        """
        生成所有用户的交互数据
        
        Args:
            user_generator: 用户生成器实例
            item_generator: 商品生成器实例
            
        Returns:
            交互数据DataFrame
        """
        print(f"正在生成交互数据...")
        
        all_interactions = []
        num_users = self.config['num_users']
        
        # 为每个用户生成交互
        for i in range(num_users):
            user_id = f'u{i+1}'
            user_interactions = self.generate_user_interactions(
                user_id, user_generator, item_generator
            )
            all_interactions.extend(user_interactions)
            
            if (i + 1) % 100 == 0:
                print(f"  已处理 {i+1}/{num_users} 个用户")
        
        # 创建DataFrame
        self.interactions_df = pd.DataFrame(all_interactions)
        
        # 添加时间戳（简化处理，不模拟真实时间序列）
        self.interactions_df['timestamp'] = np.random.randint(
            1, 1000, size=len(self.interactions_df)
        )
        
        # 重新排列列顺序
        self.interactions_df = self.interactions_df[['user_id', 'item_id', 'interaction_type', 'category', 'timestamp']]
        
        print(f"交互数据生成完成: {len(self.interactions_df)} 个交互")
        return self.interactions_df
    #endregion
    #region 获取交互统计信息  / region Get interaction statistics
    def get_interaction_statistics(self) -> Dict:
        """
        获取交互统计信息
        
        Returns:
            交互统计字典
        """
        if self.interactions_df is None:
            return {}
        
        stats = {}
        
        # 交互类型分布
        interaction_counts = self.interactions_df['interaction_type'].value_counts()
        stats['interaction_type_distribution'] = interaction_counts.to_dict()
        
        # 用户交互数量分布
        user_interaction_counts = self.interactions_df['user_id'].value_counts()
        stats['user_interaction_stats'] = {
            'min': user_interaction_counts.min(),
            'max': user_interaction_counts.max(),
            'mean': user_interaction_counts.mean(),
            'std': user_interaction_counts.std()
        }
        
        # 商品交互数量分布
        item_interaction_counts = self.interactions_df['item_id'].value_counts()
        stats['item_interaction_stats'] = {
            'min': item_interaction_counts.min(),
            'max': item_interaction_counts.max(),
            'mean': item_interaction_counts.mean(),
            'std': item_interaction_counts.std()
        }
        
        # 分类交互分布
        category_counts = self.interactions_df['category'].value_counts()
        stats['category_interaction_distribution'] = category_counts.to_dict()
        
        return stats
    #endregion
    #region 打印交互统计信息  / region Print interaction statistics 
    def print_interaction_statistics(self):
        """打印交互统计信息"""
        stats = self.get_interaction_statistics()
        
        print(f"\n=== 交互数据统计 ===")
        
        # 交互类型分布
        print(f"\n交互类型分布:")
        for interaction_type, count in stats['interaction_type_distribution'].items():
            percentage = count / len(self.interactions_df) * 100
            print(f"  {interaction_type}: {count} ({percentage:.1f}%)")
        
        # 用户交互统计
        user_stats = stats['user_interaction_stats']
        print(f"\n用户交互统计:")
        print(f"  最少交互数: {user_stats['min']}")
        print(f"  最多交互数: {user_stats['max']}")
        print(f"  平均交互数: {user_stats['mean']:.1f}")
        print(f"  标准差: {user_stats['std']:.1f}")
        
        # 商品交互统计
        item_stats = stats['item_interaction_stats']
        print(f"\n商品交互统计:")
        print(f"  最少交互数: {item_stats['min']}")
        print(f"  最多交互数: {item_stats['max']}")
        print(f"  平均交互数: {item_stats['mean']:.1f}")
        print(f"  标准差: {item_stats['std']:.1f}")
        
        # 分类交互分布（前10个）
        print(f"\n分类交互分布 (前10个):")
        category_dist = stats['category_interaction_distribution']
        sorted_categories = sorted(category_dist.items(), key=lambda x: x[1], reverse=True)
        for category, count in sorted_categories[:10]:
            percentage = count / len(self.interactions_df) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
    #endregion
    #region 获取指定用户的交互历史  / region Get user interaction history
    def get_user_interaction_history(self, user_id: str) -> pd.DataFrame:
        """
        获取指定用户的交互历史
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户交互历史DataFrame
        """
        if self.interactions_df is None:
            return pd.DataFrame()
        
        return self.interactions_df[self.interactions_df['user_id'] == user_id]
    #endregion
    #region 获取指定商品的交互历史  / region Get item interaction history
    def get_item_interaction_history(self, item_id: str) -> pd.DataFrame:
        """
        获取指定商品的交互历史
        
        Args:
            item_id: 商品ID
            
        Returns:
            商品交互历史DataFrame
        """
        if self.interactions_df is None:
            return pd.DataFrame()
        
        return self.interactions_df[self.interactions_df['item_id'] == item_id]
    #endregion
