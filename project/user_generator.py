# -*- coding: utf-8 -*-
"""
用户偏好生成模块
User Preference Generation Module
基于分类兴趣生成用户偏好数据
Generate user preference data based on category interests
"""

import numpy as np
import random
import json
import os
from typing import Dict, List, Tuple
from config import DATA_CONFIG, RANDOM_SEED

class UserGenerator:
    """用户偏好生成器 / User Preference Generator"""
    
    def __init__(self):
        """初始化用户生成器 / Initialize the user generator"""
        self.config = DATA_CONFIG
        self.user_preference_config = self.config['user_preference']
        self.category_interest_config = self.config['category_interest']
        self.num_users = self.config['num_users']
        
        # 设置随机种子 / Set random seed
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        
        # 存储生成的用户偏好 / Store generated user preferences
        self.user_preferences = {}
        
        # 加载分类受欢迎程度配置 / Load category popularity configuration
        self.category_popularity = self._load_category_popularity()

    #region 加载分类受欢迎程度配置 / Load category popularity configuration
    def _load_category_popularity(self) -> Dict[str, float]:
        """
        从外部文件加载分类受欢迎程度配置
        Load category popularity configuration from an external file
        Returns:
            分类受欢迎程度字典 / Dictionary of category popularity
            
        Raises:
            FileNotFoundError: 配置文件不存在 / Configuration file not found
            ValueError: 配置文件格式错误或缺少必要数据 / Configuration file format error or missing necessary data
            Exception: 其他加载错误 / Other loading errors
        """
        # 获取配置文件路径 / Get configuration file path
        config_file = self.category_interest_config.get('category_popularity_file', 'category_popularity.json')
        
        # 检查文件是否存在 / Check if file exists
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"分类受欢迎程度配置文件 {config_file} 不存在 / Category popularity configuration file {config_file} not found")
        
        # 加载配置文件 / Load configuration file
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # 提取分类受欢迎程度数据 / Extract category popularity data
        category_popularity = config_data.get('category_popularity', {})
        
        if not category_popularity:
            raise ValueError(f"配置文件 {config_file} 中没有找到 category_popularity 数据 / No category_popularity data found in configuration file {config_file}")
        
        print(f"成功加载分类受欢迎程度配置: {len(category_popularity)} 个分类 / Successfully loaded category popularity configuration: {len(category_popularity)} categories")
        return category_popularity
    #endregion

    #region 为指定用户分配偏好类型 / Assign preference type to specified user
    def generate_user_preference_type(self, user_id: str) -> str:
        """
        为指定用户分配偏好类型
        Assign preference type to a specified user
        Args:
            user_id: 用户ID / User ID
            
        Returns:
            偏好类型 ('single_category', 'multi_category', 'explorer') / Preference type ('single_category', 'multi_category', 'explorer')
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
    #region 基于权重随机选择分类 / Weighted random category selection
    def weighted_random_choice(self, categories: List[str], weights: Dict[str, float]) -> str:
        """
        基于权重随机选择分类
        Randomly select a category based on weights
        Args:
            categories: 分类列表 / List of categories
            weights: 分类权重字典 / Dictionary of category weights
            
        Returns:
            选中的分类 / Selected category
        """
        # 为每个分类分配权重 / Assign weights to each category
        category_weights = []
        for cat in categories:
            if cat in weights:
                category_weights.append(weights[cat])
            else:
                category_weights.append(weights.get('others', 0.01))
        
        # 归一化权重 / Normalize weights
        total_weight = sum(category_weights)
        normalized_weights = [w / total_weight for w in category_weights]
        
        # 基于权重随机选择 / Randomly select based on weights
        return np.random.choice(categories, p=normalized_weights)
    #endregion
    #region 基于权重随机采样多个分类 / Weighted random sampling of multiple categories
    def weighted_random_sample(self, categories: List[str], weights: Dict[str, float], 
                             num_samples: int) -> List[str]:
        """
        基于权重随机采样多个分类
        Randomly sample multiple categories based on weights
        Args:
            categories: 分类列表 / List of categories
            weights: 分类权重字典 / Dictionary of category weights
            num_samples: 采样数量 / Number of samples
            
        Returns:
            选中的分类列表 / List of selected categories
        """
        # 为每个分类分配权重 / Assign weights to each category
        category_weights = []
        for cat in categories:
            if cat in weights:
                category_weights.append(weights[cat])
            else:
                category_weights.append(weights.get('others', 0.01))
        
        # 归一化权重 / Normalize weights
        total_weight = sum(category_weights)
        normalized_weights = [w / total_weight for w in category_weights]
        
        # 基于权重随机采样（不重复） / Randomly sample based on weights (without replacement)
        return np.random.choice(categories, size=num_samples, replace=False, p=normalized_weights)
    #endregion
    #region 根据偏好类型为指定用户生成分类兴趣 / Generate category interests for a specified user based on preference type
    def generate_category_interests(self, user_id: str, preference_type: str, 
                                  categories: List[str]) -> Dict[str, str]:
        """
        根据偏好类型为指定用户生成分类兴趣
        Generate category interests for a specified user based on preference type
        Args:
            user_id: 用户ID / User ID
            preference_type: 偏好类型 / Preference type
            categories: 可用分类列表 / List of available categories
            
        Returns:
            分类兴趣字典 {category: interest_strength} / Category interest dictionary {category: interest_strength}
        """
        interests = {}
        category_popularity = self.category_popularity
        strength_dist = self.user_preference_config['interest_strength_distribution']
        
        if preference_type == 'single_category':
            # 单一分类用户：只对1个分类感兴趣 / Single category user: interested in only 1 category
            selected_category = self.weighted_random_choice(categories, category_popularity)
            interests[selected_category] = 'high'
            
        elif preference_type == 'multi_category':
            # 多分类用户：对2-5个分类感兴趣 / Multi-category user: interested in 2-5 categories
            min_categories = self.user_preference_config['min_categories_per_user']
            max_categories = min(self.user_preference_config['max_categories_per_user'], len(categories))
            num_categories = random.randint(min_categories, max_categories)
            
            selected_categories = self.weighted_random_sample(categories, category_popularity, num_categories)
            
            for cat in selected_categories:
                # 基于分类受欢迎程度决定兴趣强度 / Determine interest strength based on category popularity
                if random.random() < category_popularity.get(cat, 0.01):
                    strength = np.random.choice(['high', 'medium'], p=[0.6, 0.4])
                else:
                    strength = np.random.choice(['medium', 'low'], p=[0.7, 0.3])
                interests[cat] = strength
                
        else:  # explorer
            # 探索型用户：对所有分类都有兴趣，但强度不同 / Explorer user: interested in all categories, but with different strengths
            for cat in categories:
                # 基于分类受欢迎程度决定兴趣强度 / Determine interest strength based on category popularity
                if random.random() < category_popularity.get(cat, 0.01):
                    strength = np.random.choice(['high', 'medium'], p=[0.4, 0.6])
                else:
                    strength = np.random.choice(['medium', 'low'], p=[0.5, 0.5])
                interests[cat] = strength
        
        return interests
    #endregion
    #region 为所有用户生成偏好数据 / Generate preference data for all users
    def generate_all_user_preferences(self, categories: List[str]) -> Dict[str, Dict]:
        """
        为所有用户生成偏好数据
        Generate preference data for all users
        Args:
            categories: 可用分类列表 / List of available categories
            
        Returns:
            所有用户的偏好数据 / Preference data for all users
        """
        print(f"正在为 {self.num_users} 个用户生成偏好数据... / Generating preference data for {self.num_users} users...")
        
        # 统计偏好类型分布 / Count preference type distribution
        preference_type_counts = {'single_category': 0, 'multi_category': 0, 'explorer': 0}
        
        for i in range(self.num_users):
            user_id = f'u{i+1}'
            
            # 1. 确定偏好类型 / 1. Determine preference type
            preference_type = self.generate_user_preference_type(user_id)
            preference_type_counts[preference_type] += 1
            
            # 2. 生成分类兴趣 / 2. Generate category interests
            category_interests = self.generate_category_interests(user_id, preference_type, categories)
            
            # 3. 保存用户偏好 / 3. Save user preferences
            self.user_preferences[user_id] = {
                'preference_type': preference_type,
                'category_interests': category_interests
            }
        
        # 打印统计信息 / Print statistics
        print(f"用户偏好类型分布: / User Preference Type Distribution:")
        for pref_type, count in preference_type_counts.items():
            percentage = count / self.num_users * 100
            print(f"  {pref_type}: {count} 用户 ({percentage:.1f}%) /   {pref_type}: {count} users ({percentage:.1f}%)")
        
        # 统计兴趣强度分布 / Count interest strength distribution
        strength_counts = {'high': 0, 'medium': 0, 'low': 0}
        total_interests = 0
        
        for user_data in self.user_preferences.values():
            for strength in user_data['category_interests'].values():
                strength_counts[strength] += 1
                total_interests += 1
        
        print(f"\n兴趣强度分布: / \nInterest Strength Distribution:")
        for strength, count in strength_counts.items():
            percentage = count / total_interests * 100
            print(f"  {strength}: {count} 个兴趣 ({percentage:.1f}%) /   {strength}: {count} interests ({percentage:.1f}%)")
        
        return self.user_preferences
    #endregion
    #region 获取用户对特定分类商品的交互概率 / Get user interaction probability for specific category items
    def get_user_interaction_probability(self, user_id: str, item_category: str) -> float:
        """
        获取用户对特定分类商品的交互概率
        Get user interaction probability for specific category items
        Args:
            user_id: 用户ID / User ID
            item_category: 商品分类 / Item category
            
        Returns:
            交互概率 (0-1) / Interaction probability (0-1)
        """
        if user_id not in self.user_preferences:
            return 0.1  # 默认概率 / Default probability
        
        user_interests = self.user_preferences[user_id]['category_interests']
        interest_strength = user_interests.get(item_category, 'low')
        
        # 基于兴趣强度返回交互概率 / Return interaction probability based on interest strength
        interest_probs = self.category_interest_config['interest_based_probability']
        return interest_probs.get(interest_strength, 0.1)  # 默认概率为0.1 / Default probability is 0.1
    #endregion
    #region 获取用户偏好的分类列表 / Get list of user's preferred categories
    def get_user_preferred_categories(self, user_id: str) -> List[str]:
        """
        获取用户偏好的分类列表
        Get list of user's preferred categories
        Args:
            user_id: 用户ID / User ID
            
        Returns:
            偏好分类列表 / List of preferred categories
        """
        if user_id not in self.user_preferences:
            return []
        
        user_interests = self.user_preferences[user_id]['category_interests']
        # 返回所有有偏好的分类 / Return all preferred categories
        return list(user_interests.keys())
    #endregion
    #region 获取用户高兴趣的分类列表 / Get list of user's high-interest categories
    def get_user_high_interest_categories(self, user_id: str) -> List[str]:
        """
        获取用户高兴趣的分类列表
        Get list of user's high-interest categories
        Args:
            user_id: 用户ID / User ID
            
        Returns:
            高兴趣分类列表 / List of high-interest categories
        """
        if user_id not in self.user_preferences:
            return []
        
        user_interests = self.user_preferences[user_id]['category_interests']
        high_interest_categories = [cat for cat, strength in user_interests.items() 
                                  if strength == 'high']
        return high_interest_categories
    #endregion
