# -*- coding: utf-8 -*-
"""
商品分配模块
将商品分配到真实分类中，生成商品属性数据
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple
from config import DATA_CONFIG, RANDOM_SEED

class ItemGenerator:
    """商品分配生成器"""
    
    def __init__(self):
        """初始化商品生成器"""
        self.config = DATA_CONFIG
        self.enhanced_config = self.config['enhanced_processor']
        self.num_items = self.config['num_items']
        
        # 设置随机种子
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        
        # 存储生成的商品数据
        self.items_df = None
        self.category_items = {}  # 分类到商品的映射

    #region 加载分类映射数据  / region Load category mapping data
    def load_category_mapping(self, mapping_file: str) -> Dict[str, str]:
        """
        加载分类映射数据
        
        Args:
            mapping_file: 分类映射文件路径
            
        Returns:
            分类ID到名称的映射字典
        """
        import json
        with open(mapping_file, 'r', encoding='utf-8') as f:
            category_mapping = json.load(f)
        return category_mapping
    #endregion
    #region 计算分类权重  / region Calculate category weights
    def calculate_category_weights(self, categories: List[str]) -> Dict[str, float]:
        """
        计算分类权重，用于商品分配
        
        Args:
            categories: 分类列表
            
        Returns:
            分类权重字典
        """
        weights = {}
        popular_categories = self.enhanced_config['popular_categories']
        category_weights_config = self.enhanced_config['category_weights']
        
        for category in categories:
            if any(popular in category for popular in popular_categories):
                weights[category] = category_weights_config['popular']
            elif 'Baby' in category or 'Food' in category:
                weights[category] = category_weights_config['food_baby']
            else:
                weights[category] = category_weights_config['default']
        
        return weights
    #endregion
    #region 将商品分配到分类中  / region Distribute items to categories
    def distribute_items_to_categories(self, categories: List[str]) -> Dict[str, List[str]]:
        """
        将商品分配到分类中
        
        Args:
            categories: 分类列表
            
        Returns:
            分类到商品列表的映射
        """
        print(f"正在将 {self.num_items} 个商品分配到 {len(categories)} 个分类中...")
        
        # 计算分类权重
        category_weights = self.calculate_category_weights(categories)
        
        # 计算每个分类应分配的商品数量
        items_per_category = {}
        total_weight = sum(category_weights.values())
        
        # 为每个分类分配基础商品数量
        for category in categories:
            weight = category_weights[category]
            base_items = int((weight / total_weight) * self.num_items)
            
            # 应用商品数量范围限制
            min_items = self.enhanced_config['items_per_category']['min']
            max_items = self.enhanced_config['items_per_category']['max']
            
            # 热门分类应用倍数
            if any(popular in category for popular in self.enhanced_config['popular_categories']):
                max_items = int(max_items * self.enhanced_config['items_per_category']['popular_multiplier'])
            
            items_count = max(min_items, min(max_items, base_items))
            items_per_category[category] = items_count
        
        # 调整总商品数量以匹配目标
        current_total = sum(items_per_category.values())
        if current_total != self.num_items:
            # 按权重比例调整
            adjustment_factor = self.num_items / current_total
            for category in items_per_category:
                items_per_category[category] = max(
                    self.enhanced_config['items_per_category']['min'],
                    int(items_per_category[category] * adjustment_factor)
                )
        
        # 重新计算确保总数正确
        current_total = sum(items_per_category.values())
        if current_total < self.num_items:
            # 随机选择分类增加商品
            remaining = self.num_items - current_total
            categories_list = list(categories)
            for _ in range(remaining):
                category = random.choice(categories_list)
                items_per_category[category] += 1
        elif current_total > self.num_items:
            # 随机选择分类减少商品
            excess = current_total - self.num_items
            categories_list = list(categories)
            for _ in range(excess):
                category = random.choice(categories_list)
                if items_per_category[category] > self.enhanced_config['items_per_category']['min']:
                    items_per_category[category] -= 1
        
        # 生成商品ID并分配到分类
        item_id = 1
        for category, count in items_per_category.items():
            category_items = []
            for _ in range(count):
                item_id_str = f'i{item_id}'
                category_items.append(item_id_str)
                item_id += 1
            self.category_items[category] = category_items
        
        # 打印分配统计
        print(f"商品分配完成:")
        print(f"  总商品数: {sum(len(items) for items in self.category_items.values())}")
        print(f"  分类数量: {len(self.category_items)}")
        
        # 显示前10个分类的分配情况
        sorted_categories = sorted(self.category_items.items(), key=lambda x: len(x[1]), reverse=True)
        print(f"\n前10个分类的商品分配:")
        for category, items in sorted_categories[:10]:
            print(f"  {category}: {len(items)} 个商品")
        
        return self.category_items
    #endregion
    #region 生成商品属性  / region Generate item attributes
    def generate_item_attributes(self, category: str, item_id: str) -> Dict:
        """
        为商品生成属性
        
        Args:
            category: 商品分类
            item_id: 商品ID
            
        Returns:
            商品属性字典
        """
        # 获取价格范围
        price_range = self.enhanced_config['price_range']
        popularity_range = self.enhanced_config['popularity_range']
        
        # 基于分类调整价格范围
        if 'Baby' in category or 'Premium' in category:
            price_range = (price_range[0] * 1.2, price_range[1] * 1.5)
        elif 'Fresh' in category or 'Organic' in category:
            price_range = (price_range[0] * 1.1, price_range[1] * 1.3)
        elif 'Beer' in category or 'Wine' in category:
            price_range = (price_range[0] * 0.8, price_range[1] * 2.0)
        
        # 生成商品属性
        attributes = {
            'item_id': item_id,
            'category': category,
            'price': np.random.uniform(price_range[0], price_range[1]),
            'popularity_score': np.random.uniform(popularity_range[0], popularity_range[1])
        }
        
        # 基于分类调整流行度
        if any(popular in category for popular in self.enhanced_config['popular_categories']):
            attributes['popularity_score'] = min(1.0, attributes['popularity_score'] * 1.3)
        
        return attributes
    #endregion
    #region 生成商品数据DataFrame  / region Generate items dataframe
    def generate_items_dataframe(self, categories: List[str]) -> pd.DataFrame:
        """
        生成商品数据DataFrame
        
        Args:
            categories: 分类列表
            
        Returns:
            商品数据DataFrame
        """
        print("正在生成商品数据...")
        
        # 分配商品到分类
        self.distribute_items_to_categories(categories)
        
        # 生成商品数据
        items_data = []
        for category, item_ids in self.category_items.items():
            for item_id in item_ids:
                attributes = self.generate_item_attributes(category, item_id)
                items_data.append(attributes)
        
        # 创建DataFrame
        self.items_df = pd.DataFrame(items_data)
        
        # 添加分类统计信息
        self._add_category_statistics()
        
        print(f"商品数据生成完成: {len(self.items_df)} 个商品")
        return self.items_df
    #endregion
    #region 添加分类统计信息  / region Add category statistics to dataframe
    def _add_category_statistics(self):
        """添加分类统计信息到DataFrame"""
        if self.items_df is None:
            return
        
        # 计算分类统计
        category_stats = self.items_df.groupby('category').agg({
            'price': ['mean', 'std', 'min', 'max'],
            'popularity_score': ['mean', 'std', 'min', 'max'],
            'item_id': 'count'
        }).round(2)
        
        print(f"\n分类统计信息:")
        print(f"{'分类':<30} {'商品数':<8} {'平均价格':<10} {'平均流行度':<12}")
        print("-" * 70)
        
        for category in category_stats.index:
            count = category_stats.loc[category, ('item_id', 'count')]
            avg_price = category_stats.loc[category, ('price', 'mean')]
            avg_popularity = category_stats.loc[category, ('popularity_score', 'mean')]
            print(f"{category:<30} {count:<8} {avg_price:<10.2f} {avg_popularity:<12.3f}")
    #endregion
    #region 获取指定分类的商品列表  / region Get items by category
    def get_items_by_category(self, category: str) -> List[str]:
        """
        获取指定分类的商品列表
        
        Args:
            category: 分类名称
            
        Returns:
            商品ID列表
        """
        return self.category_items.get(category, [])
    #endregion
    #region 获取分类分布统计  / region Get category distribution statistics
    def get_category_distribution(self) -> Dict[str, int]:
        """
        获取分类分布统计
        
        Returns:
            分类到商品数量的映射
        """
        return {category: len(items) for category, items in self.category_items.items()}
    #endregion
    #region 获取商品信息  / region Get item info
    def get_item_info(self, item_id: str) -> Dict:
        """
        获取商品信息
        
        Args:
            item_id: 商品ID
            
        Returns:
            商品信息字典
        """
        if self.items_df is None:
            return None
        
        item_row = self.items_df[self.items_df['item_id'] == item_id]
        if item_row.empty:
            return None
        
        return item_row.iloc[0].to_dict()
    #endregion
