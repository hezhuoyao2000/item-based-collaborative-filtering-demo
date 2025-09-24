# -*- coding: utf-8 -*-
"""
商品分配模块
Item Allocation Module
将商品分配到真实分类中，生成商品属性数据
Allocate items to real categories and generate item attribute data
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple
from config import DATA_CONFIG, RANDOM_SEED

class ItemGenerator:
    """商品分配生成器 / Item Allocation Generator"""
    
    def __init__(self):
        """初始化商品生成器 / Initialize the item generator"""
        self.config = DATA_CONFIG
        self.enhanced_config = self.config['enhanced_processor']
        self.num_items = self.config['num_items']
        
        # 设置随机种子 / Set random seed
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        
        # 存储生成的商品数据 / Store generated item data
        self.items_df = None
        self.category_items = {}  # 分类到商品的映射 / Mapping from categories to items

    #region 加载分类映射数据 / Load category mapping data
    def load_category_mapping(self, mapping_file: str) -> Dict[str, str]:
        """
        加载分类映射数据
        Load category mapping data
        Args:
            mapping_file: 分类映射文件路径 / Category mapping file path
            
        Returns:
            分类ID到名称的映射字典 / Dictionary mapping category IDs to names
        """
        import json
        with open(mapping_file, 'r', encoding='utf-8') as f:
            category_mapping = json.load(f)
        return category_mapping
    #endregion
    #region 计算分类权重 / Calculate category weights
    def calculate_category_weights(self, categories: List[str]) -> Dict[str, float]:
        """
        计算分类权重，用于商品分配
        Calculate category weights for item allocation
        Args:
            categories: 分类列表 / List of categories
            
        Returns:
            分类权重字典 / Dictionary of category weights
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
    #region 将商品分配到分类中 / Distribute items to categories
    def distribute_items_to_categories(self, categories: List[str]) -> Dict[str, List[str]]:
        """
        将商品分配到分类中
        Distribute items to categories
        Args:
            categories: 分类列表 / List of categories
            
        Returns:
            分类到商品列表的映射 / Mapping from categories to item lists
        """
        print(f"正在将 {self.num_items} 个商品分配到 {len(categories)} 个分类中... / Distributing {self.num_items} items into {len(categories)} categories...")
        
        # 计算分类权重 / Calculate category weights
        category_weights = self.calculate_category_weights(categories)
        
        # 计算每个分类应分配的商品数量 / Calculate the number of items to be allocated to each category
        items_per_category = {}
        total_weight = sum(category_weights.values())
        
        # 为每个分类分配基础商品数量 / Allocate basic number of items to each category
        for category in categories:
            weight = category_weights[category]
            base_items = int((weight / total_weight) * self.num_items)
            
            # 应用商品数量范围限制 / Apply item quantity range limits
            min_items = self.enhanced_config['items_per_category']['min']
            max_items = self.enhanced_config['items_per_category']['max']
            
            # 热门分类应用倍数 / Apply multiplier for popular categories
            if any(popular in category for popular in self.enhanced_config['popular_categories']):
                max_items = int(max_items * self.enhanced_config['items_per_category']['popular_multiplier'])
            
            items_count = max(min_items, min(max_items, base_items))
            items_per_category[category] = items_count
        
        # 调整总商品数量以匹配目标 / Adjust total item count to match target
        current_total = sum(items_per_category.values())
        if current_total != self.num_items:
            # 按权重比例调整 / Adjust proportionally by weight
            adjustment_factor = self.num_items / current_total
            for category in items_per_category:
                items_per_category[category] = max(
                    self.enhanced_config['items_per_category']['min'],
                    int(items_per_category[category] * adjustment_factor)
                )
        
        # 重新计算确保总数正确 / Recalculate to ensure correct total
        current_total = sum(items_per_category.values())
        if current_total < self.num_items:
            # 随机选择分类增加商品 / Randomly select categories to add items
            remaining = self.num_items - current_total
            categories_list = list(categories)
            for _ in range(remaining):
                category = random.choice(categories_list)
                items_per_category[category] += 1
        elif current_total > self.num_items:
            # 随机选择分类减少商品 / Randomly select categories to reduce items
            excess = current_total - self.num_items
            categories_list = list(categories)
            for _ in range(excess):
                category = random.choice(categories_list)
                if items_per_category[category] > self.enhanced_config['items_per_category']['min']:
                    items_per_category[category] -= 1
        
        # 生成商品ID并分配到分类 / Generate item IDs and allocate to categories
        item_id = 1
        for category, count in items_per_category.items():
            category_items = []
            for _ in range(count):
                item_id_str = f'i{item_id}'
                category_items.append(item_id_str)
                item_id += 1
            self.category_items[category] = category_items
        
        # 打印分配统计 / Print allocation statistics
        print(f"商品分配完成: / Item allocation complete:")
        print(f"  总商品数: {sum(len(items) for items in self.category_items.values())} /   Total items: {sum(len(items) for items in self.category_items.values())}")
        print(f"  分类数量: {len(self.category_items)} /   Number of categories: {len(self.category_items)}")
        
        # 显示前10个分类的分配情况 / Display allocation for the top 10 categories
        sorted_categories = sorted(self.category_items.items(), key=lambda x: len(x[1]), reverse=True)
        print(f"\n前10个分类的商品分配: / Item Allocation for Top 10 Categories:")
        for category, items in sorted_categories[:10]:
            print(f"  {category}: {len(items)} 个商品 /   {category}: {len(items)} items")
        
        return self.category_items
    #endregion
    #region 生成商品属性 / Generate item attributes
    def generate_item_attributes(self, category: str, item_id: str) -> Dict:
        """
        为商品生成属性
        Generate attributes for an item
        Args:
            category: 商品分类 / Item category
            item_id: 商品ID / Item ID
            
        Returns:
            商品属性字典 / Dictionary of item attributes
        """
        # 获取价格范围 / Get price range
        price_range = self.enhanced_config['price_range']
        popularity_range = self.enhanced_config['popularity_range']
        
        # 基于分类调整价格范围 / Adjust price range based on category
        if 'Baby' in category or 'Premium' in category:
            price_range = (price_range[0] * 1.2, price_range[1] * 1.5)
        elif 'Fresh' in category or 'Organic' in category:
            price_range = (price_range[0] * 1.1, price_range[1] * 1.3)
        elif 'Beer' in category or 'Wine' in category:
            price_range = (price_range[0] * 0.8, price_range[1] * 2.0)
        
        # 生成商品属性 / Generate item attributes
        attributes = {
            'item_id': item_id,
            'category': category,
            'price': np.random.uniform(price_range[0], price_range[1]),
            'popularity_score': np.random.uniform(popularity_range[0], popularity_range[1])
        }
        
        # 基于分类调整流行度 / Adjust popularity based on category
        if any(popular in category for popular in self.enhanced_config['popular_categories']):
            attributes['popularity_score'] = min(1.0, attributes['popularity_score'] * 1.3)
        
        return attributes
    #endregion
    #region 生成商品数据DataFrame / Generate items dataframe
    def generate_items_dataframe(self, categories: List[str]) -> pd.DataFrame:
        """
        生成商品数据DataFrame
        Generate item data DataFrame
        Args:
            categories: 分类列表 / List of categories
            
        Returns:
            商品数据DataFrame / Item data DataFrame
        """
        print("正在生成商品数据... / Generating item data...")
        
        # 分配商品到分类 / Allocate items to categories
        self.distribute_items_to_categories(categories)
        
        # 生成商品数据 / Generate item data
        items_data = []
        for category, item_ids in self.category_items.items():
            for item_id in item_ids:
                attributes = self.generate_item_attributes(category, item_id)
                items_data.append(attributes)
        
        # 创建DataFrame / Create DataFrame
        self.items_df = pd.DataFrame(items_data)
        
        # 添加分类统计信息 / Add category statistics
        self._add_category_statistics()
        
        print(f"商品数据生成完成: {len(self.items_df)} 个商品 / Item data generation complete: {len(self.items_df)} items")
        return self.items_df
    #endregion
    #region 添加分类统计信息 / Add category statistics to dataframe
    def _add_category_statistics(self):
        """添加分类统计信息到DataFrame / Add category statistics to DataFrame"""
        if self.items_df is None:
            return
        
        # 计算分类统计 / Calculate category statistics
        category_stats = self.items_df.groupby('category').agg({
            'price': ['mean', 'std', 'min', 'max'],
            'popularity_score': ['mean', 'std', 'min', 'max'],
            'item_id': 'count'
        }).round(2)
        
        print(f"\n分类统计信息: / Category Statistics:")
        print(f"{'分类':<30} {'商品数':<8} {'平均价格':<10} {'平均流行度':<12} / {'Category':<30} {'Items':<8} {'Avg Price':<10} {'Avg Popularity':<12}")
        print("-" * 70)
        
        for category in category_stats.index:
            count = category_stats.loc[category, ('item_id', 'count')]
            avg_price = category_stats.loc[category, ('price', 'mean')]
            avg_popularity = category_stats.loc[category, ('popularity_score', 'mean')]
            print(f"{category:<30} {count:<8} {avg_price:<10.2f} {avg_popularity:<12.3f}")
    #endregion
    #region 获取指定分类的商品列表 / Get items by category
    def get_items_by_category(self, category: str) -> List[str]:
        """
        获取指定分类的商品列表
        Get list of items for a specified category
        Args:
            category: 分类名称 / Category name
            
        Returns:
            商品ID列表 / List of item IDs
        """
        return self.category_items.get(category, [])
    #endregion
    #region 获取分类分布统计 / Get category distribution statistics
    def get_category_distribution(self) -> Dict[str, int]:
        """
        获取分类分布统计
        Get category distribution statistics
        Returns:
            分类到商品数量的映射 / Mapping from categories to item counts
        """
        return {category: len(items) for category, items in self.category_items.items()}
    #endregion
    #region 获取商品信息 / Get item info
    def get_item_info(self, item_id: str) -> Dict:
        """
        获取商品信息
        Get item information
        Args:
            item_id: 商品ID / Item ID
            
        Returns:
            商品信息字典 / Dictionary of item information
        """
        if self.items_df is None:
            return None
        
        item_row = self.items_df[self.items_df['item_id'] == item_id]
        if item_row.empty:
            return None
        
        return item_row.iloc[0].to_dict()
    #endregion
