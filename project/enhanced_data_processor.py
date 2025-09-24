# -*- coding: utf-8 -*-
"""
数据预处理模块
Data Preprocessing Module
整合真实分类数据的推荐系统数据处理器
Recommendation system data processor integrating real category data
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import os
from config import DATA_CONFIG, FILE_CONFIG, RANDOM_SEED
from user_generator import UserGenerator
from item_generator import ItemGenerator
from interaction_generator import InteractionGenerator

class EnhancedDataProcessor:
    """数据处理器，支持真实分类数据 / Data processor, supporting real category data"""
    
    def __init__(self):
        """初始化数据处理器 / Initialize the data processor"""
        self.config = DATA_CONFIG
        self.file_config = FILE_CONFIG
        
        # 数据存储 / Data storage
        self.interactions_df = None
        self.items_df = None
        self.train_df = None
        self.test_df = None
        self.user_item_matrix = None
        self.user_index = None
        self.item_index = None
        
        # 分类数据 / Category data
        self.category_data = None
        self.category_mapping = None
        self.second_level_categories = []
        
        # 初始化生成器 / Initialize generators
        self.user_generator = UserGenerator()
        self.item_generator = ItemGenerator()
        self.interaction_generator = InteractionGenerator()
        
    #region 加载分类数据 / Load category data
    def load_category_data(self, mapping_file="category_mapping.json"):
        """
        加载分类数据（简化版，直接使用category_mapping.json）
        Load category data (simplified version, directly using category_mapping.json)
        Args:
            mapping_file: 分类映射文件路径 / Category mapping file path
        """
        print("正在加载分类数据... / Loading category data...")
        
        # 加载分类映射 / Load category mapping
        with open(mapping_file, 'r', encoding='utf-8') as f:
            self.category_mapping = json.load(f)
        
        # 从映射中提取分类信息 / Extract category information from mapping
        self._extract_categories_from_mapping()
        
        print(f"分类数据加载完成: {len(self.category_mapping)} 个分类 / Category data loaded: {len(self.category_mapping)} categories")
        print(f"分类数量: {len(self.second_level_categories)} / Number of categories: {len(self.second_level_categories)}")
        
        return self.category_mapping
    #endregion
    #region 加载交互数据和商品数据 / Load interactions and items data
    def load_data(self, interactions_file=None, items_file=None):
        """
        加载交互数据和商品数据
        Load interactions and items data
        Args:
            interactions_file: 交互数据文件路径 / Interactions data file path
            items_file: 商品数据文件路径 / Items data file path
        """
        print("正在加载数据... / Loading data...")
        
        # 加载交互数据 / Load interactions data
        if interactions_file and os.path.exists(interactions_file):
            self.interactions_df = pd.read_csv(interactions_file)
        else:
            raise FileNotFoundError(f"交互数据文件不存在: {interactions_file} / Interactions data file not found: {interactions_file}")
        
        # 加载商品数据 / Load items data
        if items_file and os.path.exists(items_file):
            self.items_df = pd.read_csv(items_file)
        else:
            raise FileNotFoundError(f"商品数据文件不存在: {items_file} / Items data file not found: {items_file}")
        
        # 保持时间戳为整数类型（简化处理） / Keep timestamp as integer type (simplified processing)
        # self.interactions_df['timestamp'] = pd.to_datetime(self.interactions_df['timestamp'])
        
        print(f"交互数据加载完成: {self.interactions_df.shape} / Interactions data loaded: {self.interactions_df.shape}")
        print(f"商品数据加载完成: {self.items_df.shape} / Items data loaded: {self.items_df.shape}")
        
        return self.interactions_df, self.items_df
    #endregion
    #region 数据清洗 / Clean data
    def clean_data(self):
        """
        数据清洗
        Clean data
        """
        print("正在清洗数据... / Cleaning data...")
        
        initial_size = len(self.interactions_df)
        
        # 1. 去除重复记录（同一用户对同一商品在同一时间的重复交互） / 1. Remove duplicate records (duplicate interactions of the same user with the same item at the same time)
        self.interactions_df = self.interactions_df.drop_duplicates(
            subset=['user_id', 'item_id', 'timestamp']
        )
        
        # 2. 去除无效的用户ID或商品ID / 2. Remove invalid user IDs or item IDs
        valid_users = set(self.interactions_df['user_id'].unique())
        valid_items = set(self.items_df['item_id'].unique())
        
        self.interactions_df = self.interactions_df[
            (self.interactions_df['user_id'].isin(valid_users)) &
            (self.interactions_df['item_id'].isin(valid_items))
        ]
        
        # 3. 去除交互次数过少的用户和商品（冷启动问题） / 3. Remove users and items with too few interactions (cold start problem)
        user_counts = self.interactions_df['user_id'].value_counts()
        item_counts = self.interactions_df['item_id'].value_counts()
        
        # 保留至少有2次交互的用户和商品 / Keep users and items with at least 2 interactions
        active_users = user_counts[user_counts >= 2].index
        active_items = item_counts[item_counts >= 2].index
        
        self.interactions_df = self.interactions_df[
            (self.interactions_df['user_id'].isin(active_users)) &
            (self.interactions_df['item_id'].isin(active_items))
        ]
        
        final_size = len(self.interactions_df)
        print(f"数据清洗完成: {initial_size} -> {final_size} 条记录 / Data cleaning complete: {initial_size} -> {final_size} records")
        
        return self.interactions_df
    #endregion
    #region 划分训练集和测试集 / Split train and test sets
    def split_train_test(self):
        """
        按时间划分训练集和测试集
        Split training and test sets by time
        """
        print("正在划分训练集和测试集... / Splitting training and test sets...")
        
        # 按时间排序 / Sort by time
        self.interactions_df = self.interactions_df.sort_values('timestamp')
        
        # 按时间比例划分（前80%作为训练集，后20%作为测试集） / Split by time ratio (first 80% as training set, last 20% as test set)
        train_ratio = self.config['train_ratio']
        split_point = int(len(self.interactions_df) * train_ratio)
        
        self.train_df = self.interactions_df.iloc[:split_point].copy()
        self.test_df = self.interactions_df.iloc[split_point:].copy()
        
        print(f"训练集大小: {len(self.train_df)} / Training set size: {len(self.train_df)}")
        print(f"测试集大小: {len(self.test_df)} / Test set size: {len(self.test_df)}")
        print(f"训练集时间范围: {self.train_df['timestamp'].min()} - {self.train_df['timestamp'].max()} / Training set time range: {self.train_df['timestamp'].min()} - {self.train_df['timestamp'].max()}")
        print(f"测试集时间范围: {self.test_df['timestamp'].min()} - {self.test_df['timestamp'].max()} / Test set time range: {self.test_df['timestamp'].min()} - {self.test_df['timestamp'].max()}")
        
        return self.train_df, self.test_df
    #endregion
    #region 获取指定用户的交互记录 / Get user interactions
    def get_user_interactions(self, user_id, data_type='train'):
        """
        获取指定用户的交互记录
        Get specified user's interaction records
        Args:
            user_id: 用户ID / User ID
            data_type: 数据类型 ('train', 'test', 'all') / Data type ('train', 'test', 'all')
        
        Returns:
            用户交互记录DataFrame / User interaction records DataFrame
        """
        if data_type == 'train':
            df = self.train_df
        elif data_type == 'test':
            df = self.test_df
        else:
            df = self.interactions_df
        
        return df[df['user_id'] == user_id]
    #endregion
    #region 获取数据统计信息 / Get statistics
    def get_statistics(self):
        """
        获取数据统计信息
        Get data statistics
        """
        print("\n=== 数据统计信息 / Data Statistics ===")
        
        # 基本统计 / Basic statistics
        print(f"总用户数: {len(self.interactions_df['user_id'].unique())} / Total users: {len(self.interactions_df['user_id'].unique())}")
        print(f"总商品数: {len(self.interactions_df['item_id'].unique())} / Total items: {len(self.interactions_df['item_id'].unique())}")
        print(f"总交互数: {len(self.interactions_df)} / Total interactions: {len(self.interactions_df)}")
        
        # 交互类型分布 / Interaction type distribution
        interaction_counts = self.interactions_df['interaction_type'].value_counts()
        print(f"\n交互类型分布: / Interaction Type Distribution:")
        for interaction_type, count in interaction_counts.items():
            percentage = count / len(self.interactions_df) * 100
            print(f"  {interaction_type}: {count} ({percentage:.1f}%) /   {interaction_type}: {count} ({percentage:.1f}%)")
        
        # 用户交互次数统计 / User interaction count statistics
        user_interaction_counts = self.interactions_df['user_id'].value_counts()
        print(f"\n用户交互次数统计: / User Interaction Count Statistics:")
        print(f"  平均交互次数: {user_interaction_counts.mean():.2f} /   Average interactions: {user_interaction_counts.mean():.2f}")
        print(f"  中位数交互次数: {user_interaction_counts.median():.2f} /   Median interactions: {user_interaction_counts.median():.2f}")
        print(f"  最大交互次数: {user_interaction_counts.max()} /   Maximum interactions: {user_interaction_counts.max()}")
        print(f"  最小交互次数: {user_interaction_counts.min()} /   Minimum interactions: {user_interaction_counts.min()}")
        
        # 商品交互次数统计 / Item interaction count statistics
        item_interaction_counts = self.interactions_df['item_id'].value_counts()
        print(f"\n商品交互次数统计: / Item Interaction Count Statistics:")
        print(f"  平均交互次数: {item_interaction_counts.mean():.2f} /   Average interactions: {item_interaction_counts.mean():.2f}")
        print(f"  中位数交互次数: {item_interaction_counts.median():.2f} /   Median interactions: {item_interaction_counts.median():.2f}")
        print(f"  最大交互次数: {item_interaction_counts.max()} /   Maximum interactions: {item_interaction_counts.max()}")
        print(f"  最小交互次数: {item_interaction_counts.min()} /   Minimum interactions: {item_interaction_counts.min()}")
        
        return {
            'total_users': len(self.interactions_df['user_id'].unique()),
            'total_items': len(self.interactions_df['item_id'].unique()),
            'total_interactions': len(self.interactions_df),
            'interaction_type_distribution': interaction_counts.to_dict(),
            'user_interaction_stats': {
                'mean': user_interaction_counts.mean(),
                'median': user_interaction_counts.median(),
                'max': user_interaction_counts.max(),
                'min': user_interaction_counts.min()
            },
            'item_interaction_stats': {
                'mean': item_interaction_counts.mean(),
                'median': item_interaction_counts.median(),
                'max': item_interaction_counts.max(),
                'min': item_interaction_counts.min()
            }
        }
    #endregion
    #region 从分类映射中提取分类信息 / Extract categories from mapping
    def _extract_categories_from_mapping(self):
        """从分类映射中提取分类信息 / Extract category information from category mapping"""
        self.second_level_categories = []
        
        for category_id, category_name in self.category_mapping.items():
            # 简化处理：所有分类都作为第二层分类 / Simplified processing: all categories are treated as second-level categories
            # 可以根据分类名称推断父分类 / Parent categories can be inferred from category names
            parent_name = self._infer_parent_category(category_name)
            
            self.second_level_categories.append({
                'id': category_id,
                'name': category_name,
                'parent_id': f"parent_{parent_name.replace(' ', '_').lower()}",
                'parent_name': parent_name
            })
    #endregion
    #region 根据分类名称推断父分类 / Infer parent category
    def _infer_parent_category(self, category_name):
        """根据分类名称推断父分类 / Infer parent category based on category name"""
        # 基于分类名称推断父分类 / Infer parent category based on category name
        if 'Baby' in category_name or 'Toddler' in category_name:
            return 'Baby & Toddler'
        elif 'Fresh' in category_name or 'Bakery' in category_name:
            return 'Fresh Foods & Bakery'
        elif 'Health' in category_name or 'Body' in category_name:
            return 'Health & Body'
        elif 'Pantry' in category_name:
            return 'Pantry'
        elif 'Household' in category_name or 'Cleaning' in category_name:
            return 'Household & Cleaning'
        elif 'Drinks' in category_name or 'Coffee' in category_name or 'Tea' in category_name:
            return 'Hot & Cold Drinks'
        elif 'Beer' in category_name or 'Wine' in category_name:
            return 'Beer & Wine'
        elif 'Pet' in category_name:
            return 'Pet Care'
        else:
            return 'Other'
    #endregion
    #region 使用真实分类数据生成商品数据 / Generate items with real categories
    def generate_items_with_real_categories(self, num_items=None):
        """
        使用真实分类数据生成商品数据（新版本）
        Generate item data using real category data (new version)
        Args:
            num_items: 商品数量，如果为None则使用配置文件中的数量 / Number of items, if None, use the quantity in the configuration file
        
        Returns:
            包含真实分类信息的商品DataFrame / Item DataFrame containing real category information
        """
        if num_items is None:
            num_items = self.config['num_items']
        
        if self.second_level_categories is None:
            raise ValueError("请先加载分类数据 / Please load category data first")
        
        print(f"正在生成 {num_items} 个商品，使用真实分类数据... / Generating {num_items} items using real category data...")
        
        # 提取分类名称 / Extract category names
        category_names = [cat['name'] for cat in self.second_level_categories]
        
        # 使用新的商品生成器 / Use the new item generator
        items_df = self.item_generator.generate_items_dataframe(category_names)
        
        # 添加分类ID和父分类信息 / Add category ID and parent category information
        items_df = self._add_category_metadata(items_df)
        
        return items_df
    #endregion
    #region 为商品数据添加分类元数据 / Add category metadata to items dataframe
    def _add_category_metadata(self, items_df):
        """
        为商品数据添加分类元数据
        Add category metadata to item data
        Args:
            items_df: 商品DataFrame / Item DataFrame
            
        Returns:
            添加了分类元数据的DataFrame / DataFrame with added category metadata
        """
        # 创建分类名称到分类信息的映射 / Create a mapping from category names to category information
        category_info_map = {cat['name']: cat for cat in self.second_level_categories}
        
        # 添加分类ID和父分类信息 / Add category ID and parent category information
        items_df['category_id'] = items_df['category'].map(lambda x: category_info_map[x]['id'])
        items_df['parent_category'] = items_df['category'].map(lambda x: category_info_map[x]['parent_name'])
        items_df['parent_category_id'] = items_df['category'].map(lambda x: category_info_map[x]['parent_id'])
        
        return items_df
    #endregion
    #region 为分类生成权重 / Generate category weights
    def _generate_category_weights(self, category_names):
        """
        为分类生成权重，模拟真实分布
        Generate weights for categories to simulate real distribution
        Args:
            category_names: 分类名称列表 / List of category names
        
        Returns:
            权重数组 / Weight array
        """
        # 从配置文件获取热门分类和权重设置 / Get popular categories and weight settings from the configuration file
        enhanced_config = self.config.get('enhanced_processor', {})
        popular_categories = enhanced_config.get('popular_categories', [
            'Baby & Toddler Food',
            'Health & Body',
            'Household & Cleaning',
            'Pantry',
            'Fresh Foods & Bakery',
            'Hot & Cold Drinks'
        ])
        
        category_weights = enhanced_config.get('category_weights', {
            'popular': 3.0,
            'food_baby': 2.0,
            'default': 1.0
        })
        
        weights = np.ones(len(category_names))
        
        # 为热门分类分配更高权重 / Assign higher weights to popular categories
        for i, cat_name in enumerate(category_names):
            if any(popular in cat_name for popular in popular_categories):
                weights[i] = category_weights.get('popular', 3.0)
            elif 'Baby' in cat_name or 'Food' in cat_name:
                weights[i] = category_weights.get('food_baby', 2.0)
            else:
                weights[i] = category_weights.get('default', 1.0)
        
        # 归一化权重 / Normalize weights
        weights = weights / weights.sum()
        return weights
    #endregion
    #region 创建增强版用户-物品交互矩阵 / Create enhanced user-item interaction matrix
    def create_enhanced_user_item_matrix(self, data_type='train'):
        """
        创建增强版用户-物品交互矩阵，包含分类信息
        Create an enhanced user-item interaction matrix, including category information
        Args:
            data_type: 数据类型 ('train', 'test', 'all') / Data type ('train', 'test', 'all')
        
        Returns:
            用户-物品交互矩阵 / User-item interaction matrix
        """
        print(f"正在创建增强版{data_type}用户-物品交互矩阵... / Creating enhanced {data_type} user-item interaction matrix...")
        
        # 选择数据集 / Select dataset
        if data_type == 'train':
            df = self.train_df
        elif data_type == 'test':
            df = self.test_df
        else:
            df = self.interactions_df
        
        # 获取所有用户和商品 / Get all users and items
        users = sorted(df['user_id'].unique())
        items = sorted(df['item_id'].unique())
        
        # 创建用户和商品的索引映射 / Create index mappings for users and items
        self.user_index = {user: idx for idx, user in enumerate(users)}
        self.item_index = {item: idx for idx, item in enumerate(items)}
        
        # 创建交互权重映射 / Create interaction weight mapping
        interaction_weights = self.config['interaction_weights']
        
        # 计算每个用户-商品对的权重（累加多次交互） / Calculate the weight for each user-item pair (accumulate multiple interactions)
        user_item_weights = {}
        for _, row in df.iterrows():
            user = row['user_id']
            item = row['item_id']
            interaction_type = row['interaction_type']
            weight = interaction_weights.get(interaction_type, 1)
            
            key = (user, item)
            if key in user_item_weights:
                user_item_weights[key] += weight
            else:
                user_item_weights[key] = weight
        
        # 创建稀疏矩阵 / Create sparse matrix
        rows = []
        cols = []
        data = []
        
        for (user, item), weight in user_item_weights.items():
            rows.append(self.user_index[user])
            cols.append(self.item_index[item])
            data.append(weight)
        
        # 创建CSR稀疏矩阵 / Create CSR sparse matrix
        matrix = csr_matrix((data, (rows, cols)), 
                          shape=(len(users), len(items)))
        
        if data_type == 'train':
            self.user_item_matrix = matrix
        
        print(f"矩阵形状: {matrix.shape} / Matrix shape: {matrix.shape}")
        print(f"非零元素数量: {matrix.nnz} / Number of non-zero elements: {matrix.nnz}")
        print(f"矩阵密度: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4f} / Matrix density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4f}")
        
        return matrix, users, items
    #endregion
    #region 获取分类统计信息 / Get category statistics
    def get_category_statistics(self):
        """
        获取分类统计信息
        Get category statistics
        Returns:
            分类统计信息字典 / Dictionary of category statistics
        """
        if self.items_df is None:
            print("请先加载商品数据 / Please load item data first")
            return None
        
        print("\n=== 分类统计信息 / Category Statistics ===")
        
        # 第二层分类分布 / Second-level category distribution
        second_level_counts = self.items_df['category'].value_counts()
        print(f"\n第二层分类分布 (前10个): / Second-level Category Distribution (Top 10):")
        for category, count in second_level_counts.head(10).items():
            print(f"  {category}: {count} 个商品 /   {category}: {count} items")
        
        # 第一层分类分布 / First-level category distribution
        parent_counts = self.items_df['parent_category'].value_counts()
        print(f"\n第一层分类分布: / First-level Category Distribution:")
        for category, count in parent_counts.items():
            print(f"  {category}: {count} 个商品 /   {category}: {count} items")
        
        return {
            'second_level_distribution': second_level_counts.to_dict(),
            'parent_level_distribution': parent_counts.to_dict(),
            'total_categories': len(second_level_counts),
            'total_parent_categories': len(parent_counts)
        }
    #endregion
    #region 生成完整的数据集 / Generate complete dataset
    def generate_complete_dataset(self, mapping_file=None):
        """
        生成完整的数据集（用户偏好 + 商品数据 + 交互数据）
        Generate a complete dataset (user preferences + item data + interaction data)
        Args:
            mapping_file: 分类映射文件路径 / Category mapping file path
        
        Returns:
            完整的数据集字典 / Dictionary of the complete dataset
        """
        print("开始生成完整数据集... / Starting to generate complete dataset...")
        
        # 1. 加载分类数据 / 1. Load category data
        if mapping_file is None:
            mapping_file = self.config['category_mapping_file']
        
        self.load_category_data(mapping_file)
        
        # 2. 生成用户偏好 / 2. Generate user preferences
        print("\n步骤1: 生成用户偏好... / Step 1: Generating user preferences...")
        category_names = [cat['name'] for cat in self.second_level_categories]
        user_preferences = self.user_generator.generate_all_user_preferences(category_names)
        
        # 3. 生成商品数据 / 3. Generate item data
        print("\n步骤2: 生成商品数据... / Step 2: Generating item data...")
        self.items_df = self.item_generator.generate_items_dataframe(category_names)
        self.items_df = self._add_category_metadata(self.items_df)
        
        # 4. 生成交互数据 / 4. Generate interaction data
        print("\n步骤3: 生成交互数据... / Step 3: Generating interaction data...")
        self.interactions_df = self.interaction_generator.generate_all_interactions(
            self.user_generator, self.item_generator
        )
        
        # 5. 数据预处理 / 5. Data preprocessing
        print("\n步骤4: 数据预处理... / Step 4: Data preprocessing...")
        self.clean_data()
        self.split_train_test()
        self.create_enhanced_user_item_matrix('train')
        
        # 6. 显示统计信息 / 6. Display statistics
        print("\n步骤5: 生成统计信息... / Step 5: Generating statistics...")
        self.get_statistics()
        self.get_category_statistics()
        self.interaction_generator.print_interaction_statistics()
        
        # 7. 保存数据到文件 / 7. Save data to file
        print("\n步骤6: 保存数据到文件... / Step 6: Saving data to file...")
        self._save_generated_data()
        
        print("\n完整数据集生成完成! / Complete dataset generation finished!")
        
        return {
            'interactions_df': self.interactions_df,
            'train_df': self.train_df,
            'test_df': self.test_df,
            'user_item_matrix': self.user_item_matrix,
            'user_index': self.user_index,
            'item_index': self.item_index,
            'items_df': self.items_df,
            'category_data': self.category_data,
            'category_mapping': self.category_mapping,
            'user_preferences': user_preferences
        }
    #endregion
    #region 保存生成的数据到文件 / Save generated data to file
    def _save_generated_data(self):
        """保存生成的数据到文件 / Save generated data to file"""
        import os
        
        # 确保data目录存在 / Ensure data directory exists
        data_dir = self.file_config['data_dir']
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # 保存交互数据 / Save interactions data
        interactions_file = os.path.join(data_dir, 'interactions.csv')
        self.interactions_df.to_csv(interactions_file, index=False)
        print(f"交互数据已保存到: {interactions_file} / Interactions data saved to: {interactions_file}")
        
        # 保存商品数据 / Save items data
        items_file = os.path.join(data_dir, 'items.csv')
        self.items_df.to_csv(items_file, index=False)
        print(f"商品数据已保存到: {items_file} / Items data saved to: {items_file}")
        
        # 保存训练集和测试集 / Save training and test sets
        train_file = os.path.join(data_dir, 'train_interactions.csv')
        test_file = os.path.join(data_dir, 'test_interactions.csv')
        self.train_df.to_csv(train_file, index=False)
        self.test_df.to_csv(test_file, index=False)
        print(f"训练集已保存到: {train_file} / Training set saved to: {train_file}")
        print(f"测试集已保存到: {test_file} / Test set saved to: {test_file}")
        
        # 保存用户偏好数据 / Save user preferences data
        import json
        user_preferences_file = os.path.join(data_dir, 'user_preferences.json')
        with open(user_preferences_file, 'w', encoding='utf-8') as f:
            json.dump(self.user_generator.user_preferences, f, ensure_ascii=False, indent=2)
        print(f"用户偏好数据已保存到: {user_preferences_file} / User preferences data saved to: {user_preferences_file}")
        
        # 保存分类统计信息 / Save category statistics
        category_stats = self.get_category_statistics()
        stats_file = os.path.join(data_dir, 'category_statistics.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(category_stats, f, ensure_ascii=False, indent=2)
        print(f"分类统计信息已保存到: {stats_file} / Category statistics saved to: {stats_file}")
    #endregion
    #region 执行完整的数据预处理流程 / Process all
    def process_all(self, interactions_file=None, items_file=None, 
                   category_file=None, mapping_file=None):
        """
        执行完整的数据预处理流程（使用真实分类数据）
        Execute the complete data preprocessing pipeline (using real category data)
        Args:
            interactions_file: 交互数据文件路径 / Interactions data file path
            items_file: 商品数据文件路径（如果为None则生成新数据） / Items data file path (if None, new data will be generated)
            category_file: 分类数据文件路径（如果为None则使用配置文件中的路径） / Category data file path (if None, use path from config file)
            mapping_file: 分类映射文件路径（如果为None则使用配置文件中的路径） / Category mapping file path (if None, use path from config file)
        
        Returns:
            处理后的数据字典 / Dictionary of processed data
        """
        print("开始数据预处理... / Starting data preprocessing...")
        
        # 1. 加载分类数据 / 1. Load category data
        if category_file is None:
            category_file = self.config['category_file']
        if mapping_file is None:
            mapping_file = self.config['category_mapping_file']
        
        self.load_category_data(category_file, mapping_file)
        
        # 2. 加载或生成商品数据 / 2. Load or generate item data
        if items_file is not None and os.path.exists(items_file):
            print("加载现有商品数据... / Loading existing item data...")
            self.items_df = pd.read_csv(items_file)
        else:
            print("生成新的商品数据... / Generating new item data...")
            self.items_df = self.generate_items_with_real_categories()
            # 保存生成的商品数据 / Save generated item data
            if items_file is None:
                items_file = os.path.join(self.file_config['data_dir'], 'items.csv')
            self.items_df.to_csv(items_file, index=False)
            print(f"商品数据已保存到: {items_file} / Item data saved to: {items_file}")
        
        # 3. 加载交互数据 / 3. Load interactions data
        self.load_data(interactions_file, items_file)
        
        # 4. 清洗数据 / 4. Clean data
        self.clean_data()
        
        # 5. 划分训练集和测试集 / 5. Split training and test sets
        self.split_train_test()
        
        # 6. 创建用户-物品交互矩阵 / 6. Create user-item interaction matrix
        self.create_enhanced_user_item_matrix('train')
        
        # 7. 显示统计信息 / 7. Display statistics
        self.get_statistics()
        self.get_category_statistics()
        
        print("数据预处理完成! / Data preprocessing complete!")
        
        return {
            'interactions_df': self.interactions_df,
            'train_df': self.train_df,
            'test_df': self.test_df,
            'user_item_matrix': self.user_item_matrix,
            'user_index': self.user_index,
            'item_index': self.item_index,
            'items_df': self.items_df,
            'category_data': self.category_data,
            'category_mapping': self.category_mapping
        }
    #endregion
