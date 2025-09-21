# -*- coding: utf-8 -*-
"""
用户偏好生成模块
基于分类兴趣生成用户偏好数据
"""

import numpy as np
import random
from typing import Dict, List, Tuple
from config import DATA_CONFIG, RANDOM_SEED

class UserGenerator:
    """用户偏好生成器"""
    
    def __init__(self):
        """初始化用户生成器"""
        self.config = DATA_CONFIG
        self.user_preference_config = self.config['user_preference']
        self.category_interest_config = self.config['category_interest']
        self.num_users = self.config['num_users']
        
        # 设置随机种子
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        
        # 存储生成的用户偏好
        self.user_preferences = {}

    #region 为指定用户分配偏好类型
    def generate_user_preference_type(self, user_id: str) -> str:
        """
        为指定用户分配偏好类型
        
        Args:
            user_id: 用户ID
            
        Returns:
            偏好类型 ('single_category', 'multi_category', 'explorer')
        """
        random_value = random.random()
        preference_types = self.user_preference_config['preference_types']
        
        if random_value < preference_types['single_category']:
            return 'single_category'
        elif random_value < preference_types['single_category'] + preference_types['multi_category']:
            return 'multi_category'
        else:
            return 'explorer'
    #endregion
    #region 基于权重随机选择分类
    def weighted_random_choice(self, categories: List[str], weights: Dict[str, float]) -> str:
        """
        基于权重随机选择分类
        
        Args:
            categories: 分类列表
            weights: 分类权重字典
            
        Returns:
            选中的分类
        """
        # 为每个分类分配权重
        category_weights = []
        for cat in categories:
            if cat in weights:
                category_weights.append(weights[cat])
            else:
                category_weights.append(weights.get('others', 0.01))
        
        # 归一化权重
        total_weight = sum(category_weights)
        normalized_weights = [w / total_weight for w in category_weights]
        
        # 基于权重随机选择
        return np.random.choice(categories, p=normalized_weights)
    #endregion
    #region 基于权重随机采样多个分类
    def weighted_random_sample(self, categories: List[str], weights: Dict[str, float], 
                             num_samples: int) -> List[str]:
        """
        基于权重随机采样多个分类
        
        Args:
            categories: 分类列表
            weights: 分类权重字典
            num_samples: 采样数量
            
        Returns:
            选中的分类列表
        """
        # 为每个分类分配权重
        category_weights = []
        for cat in categories:
            if cat in weights:
                category_weights.append(weights[cat])
            else:
                category_weights.append(weights.get('others', 0.01))
        
        # 归一化权重
        total_weight = sum(category_weights)
        normalized_weights = [w / total_weight for w in category_weights]
        
        # 基于权重随机采样（不重复）
        return np.random.choice(categories, size=num_samples, replace=False, p=normalized_weights)
    #endregion
    #region 根据偏好类型为指定用户生成分类兴趣
    def generate_category_interests(self, user_id: str, preference_type: str, 
                                  categories: List[str]) -> Dict[str, str]:
        """
        根据偏好类型为指定用户生成分类兴趣
        
        Args:
            user_id: 用户ID
            preference_type: 偏好类型
            categories: 可用分类列表
            
        Returns:
            分类兴趣字典 {category: interest_strength}
        """
        interests = {}
        category_popularity = self.category_interest_config['category_popularity']
        strength_dist = self.user_preference_config['interest_strength_distribution']
        
        if preference_type == 'single_category':
            # 单一分类用户：只对1个分类感兴趣
            selected_category = self.weighted_random_choice(categories, category_popularity)
            interests[selected_category] = 'high'
            
        elif preference_type == 'multi_category':
            # 多分类用户：对2-5个分类感兴趣
            min_categories = self.user_preference_config['min_categories_per_user']
            max_categories = min(self.user_preference_config['max_categories_per_user'], len(categories))
            num_categories = random.randint(min_categories, max_categories)
            
            selected_categories = self.weighted_random_sample(categories, category_popularity, num_categories)
            
            for cat in selected_categories:
                # 基于分类受欢迎程度决定兴趣强度
                if random.random() < category_popularity.get(cat, 0.01):
                    strength = np.random.choice(['high', 'medium'], p=[0.6, 0.4])
                else:
                    strength = np.random.choice(['medium', 'low'], p=[0.7, 0.3])
                interests[cat] = strength
                
        else:  # explorer
            # 探索型用户：对所有分类都有兴趣，但强度不同
            for cat in categories:
                # 基于分类受欢迎程度决定兴趣强度
                if random.random() < category_popularity.get(cat, 0.01):
                    strength = np.random.choice(['high', 'medium'], p=[0.4, 0.6])
                else:
                    strength = np.random.choice(['medium', 'low'], p=[0.5, 0.5])
                interests[cat] = strength
        
        return interests
    #endregion
    #region 为所有用户生成偏好数据
    def generate_all_user_preferences(self, categories: List[str]) -> Dict[str, Dict]:
        """
        为所有用户生成偏好数据
        
        Args:
            categories: 可用分类列表
            
        Returns:
            所有用户的偏好数据
        """
        print(f"正在为 {self.num_users} 个用户生成偏好数据...")
        
        # 统计偏好类型分布
        preference_type_counts = {'single_category': 0, 'multi_category': 0, 'explorer': 0}
        
        for i in range(self.num_users):
            user_id = f'u{i+1}'
            
            # 1. 确定偏好类型
            preference_type = self.generate_user_preference_type(user_id)
            preference_type_counts[preference_type] += 1
            
            # 2. 生成分类兴趣
            category_interests = self.generate_category_interests(user_id, preference_type, categories)
            
            # 3. 保存用户偏好
            self.user_preferences[user_id] = {
                'preference_type': preference_type,
                'category_interests': category_interests
            }
        
        # 打印统计信息
        print(f"用户偏好类型分布:")
        for pref_type, count in preference_type_counts.items():
            percentage = count / self.num_users * 100
            print(f"  {pref_type}: {count} 用户 ({percentage:.1f}%)")
        
        # 统计兴趣强度分布
        strength_counts = {'high': 0, 'medium': 0, 'low': 0}
        total_interests = 0
        
        for user_data in self.user_preferences.values():
            for strength in user_data['category_interests'].values():
                strength_counts[strength] += 1
                total_interests += 1
        
        print(f"\n兴趣强度分布:")
        for strength, count in strength_counts.items():
            percentage = count / total_interests * 100
            print(f"  {strength}: {count} 个兴趣 ({percentage:.1f}%)")
        
        return self.user_preferences
    #endregion
    #region 获取用户对特定分类商品的交互概率
    def get_user_interaction_probability(self, user_id: str, item_category: str) -> float:
        """
        获取用户对特定分类商品的交互概率
        
        Args:
            user_id: 用户ID
            item_category: 商品分类
            
        Returns:
            交互概率 (0-1)
        """
        if user_id not in self.user_preferences:
            return 0.1  # 默认概率
        
        user_interests = self.user_preferences[user_id]['category_interests']
        interest_strength = user_interests.get(item_category, 'low')
        
        # 基于兴趣强度返回交互概率
        interest_probs = self.category_interest_config['interest_based_probability']
        return interest_probs.get(interest_strength, 0.1)  # 默认概率为0.1
    #endregion
    #region 获取用户偏好的分类列表
    def get_user_preferred_categories(self, user_id: str) -> List[str]:
        """
        获取用户偏好的分类列表
        
        Args:
            user_id: 用户ID
            
        Returns:
            偏好分类列表
        """
        if user_id not in self.user_preferences:
            return []
        
        user_interests = self.user_preferences[user_id]['category_interests']
        # 返回所有有偏好的分类
        return list(user_interests.keys())
    #endregion
    #region 获取用户高兴趣的分类列表
    def get_user_high_interest_categories(self, user_id: str) -> List[str]:
        """
        获取用户高兴趣的分类列表
        
        Args:
            user_id: 用户ID
            
        Returns:
            高兴趣分类列表
        """
        if user_id not in self.user_preferences:
            return []
        
        user_interests = self.user_preferences[user_id]['category_interests']
        high_interest_categories = [cat for cat, strength in user_interests.items() 
                                  if strength == 'high']
        return high_interest_categories
    #endregion
