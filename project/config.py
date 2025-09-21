# -*- coding: utf-8 -*-
"""
Recommendation System Configuration File
推荐系统配置文件
Centralized management of all parameters for easy adjustment and experimentation
集中管理所有参数，便于调整和实验
"""

# 数据生成参数 / Data Generation Parameters
DATA_CONFIG = {
    # 基础数据量参数 / region Basic Data Volume Parameters
    'num_users': 1000,           # Number of users / 用户数量
    'num_items': 3000,           # Number of items / 商品数量
    'num_interactions': 150000,  # Total number of interactions / 总交互数量
    'interactions_per_user_range': (50, 300),  # 每用户交互数量范围
    'interactions_per_item_range': (10, 500),  # 每商品交互数量范围
    #endregion

    # region 商品类别配置 / region Item Category Configuration
    # 使用真实分类数据，从category_mapping.json加载
    'category_mapping_file': 'category_mapping.json',  # 分类映射文件
    #endregion

    # region 交互类型配置 / region Interaction Type Configuration
    'interaction_types': {
        'click': 0.7,     # Click probability / 点击概率
        'cart': 0.2,      # Add to cart probability / 加购概率
        'favorite': 0.1   # Favorite probability / 收藏概率
    },
    #endregion
    # region 交互权重配置（用于计算用户-物品交互矩阵） / region Interaction Weight Configuration (for calculating user-item interaction matrix)
    'interaction_weights': {
        'click': 1,
        'cart': 3,
        'favorite': 5
    },
    #endregion
    # region 时间配置 / region Time Configuration
    'simulation_days': 30,  # Time span for simulated data (days) / 模拟数据的时间跨度（天）
    'train_ratio': 0.8,     # Training set ratio / 训练集比例
    #endregion
    
    # region 增强版数据处理器配置 / region Enhanced Data Processor Configuration
    'enhanced_processor': {
        'popular_categories': [  # 热门分类，分配更高权重
            'Baby & Toddler Food',
            'Health & Body',
            'Household & Cleaning',
            'Pantry',
            'Fresh Foods & Bakery',
            'Hot & Cold Drinks'
        ],
        'category_weights': {
            'popular': 3.0,      # 热门分类权重
            'food_baby': 2.0,    # 食品和婴儿用品权重
            'default': 1.0       # 默认权重
        },
        'price_range': (10, 500),  # 价格范围
        'popularity_range': (0, 1),  # 流行度范围
        'items_per_category': {
            'min': 5,                # 每个分类最少商品数
            'max': 20,               # 每个分类最多商品数
            'popular_multiplier': 2  # 热门分类商品数倍数
        }
    },
    #endregion
    
    # region 用户偏好配置 / region User Preference Configuration
    'user_preference': {
        'preference_types': {
            'single_category': 0.3,      # 30%用户只对单一分类感兴趣
            'multi_category': 0.65,       # 65%用户对多个分类感兴趣
            'explorer': 0.05              # 5%用户对所有分类都有兴趣
        },
        'max_categories_per_user': 5,    # 每用户最多感兴趣的分类数
        'min_categories_per_user': 1,    # 每用户最少感兴趣的分类数
        'interest_strength_distribution': {
            'high': 0.3,                 # 30%高兴趣
            'medium': 0.5,               # 50%中等兴趣
            'low': 0.2                   # 20%低兴趣
        }
    },
    #endregion
    
    # region 分类兴趣配置 / region Category Interest Configuration
    'category_interest': {
        'category_popularity_file': 'category_popularity.json',  # 分类受欢迎程度配置文件
        'interest_based_probability': {
            'high': 0.8,        # 高兴趣分类的交互概率
            'medium': 0.4,      # 中等兴趣分类的交互概率
            'low': 0.1          # 低兴趣分类的交互概率
        }
    },
    #endregion
}

# region 模型参数 / region Model Parameters
MODEL_CONFIG = {
    'similarity_metric': 'pearson',  # Similarity calculation method / 相似度计算方法
    'min_similarity': 0.05,          # Minimum similarity threshold / 最小相似度阈值
    'top_n': 10,                    # Number of recommended items / 推荐商品数量
    # 可选的相似度方法: 'cosine', 'pearson', 'adjusted_cosine'
    # Available similarity methods: 'cosine', 'pearson', 'adjusted_cosine'
}

# region 相似度算法参数 / region Similarity Algorithm Parameters
SIMILARITY_CONFIG = {
    # 皮尔逊相关系数参数
    'pearson': {
        'min_common_users': 2,      # 最小共同用户数
        'handle_nan': True,         # 处理NaN值
    },
    
    # 修正余弦相似度参数
    'adjusted_cosine': {
        'min_common_users': 1,      # 最小共同用户数
        'use_user_mean': True,      # 使用用户平均评分
    },
    
    
    # 余弦相似度参数
    'cosine': {
        'normalize': True,          # 是否归一化
        'sparse_optimization': True, # 稀疏矩阵优化
    },
    
    # 通用参数
    'common': {
        'min_similarity': 0.1,      # 最小相似度阈值
        'symmetric': True,          # 是否确保矩阵对称
        'zero_diagonal': True,      # 是否将对角线设为0
    }
}

# region 评估参数 / region Evaluation Parameters
EVALUATION_CONFIG = {
    'metrics': ['precision', 'recall', 'f1_score'],
    'k_values': [5, 10, 20],  # Top-K values for evaluation / 评估的Top-K值
}

# region 文件路径配置 / region File Path Configuration
FILE_CONFIG = {
    'interactions_file': 'interactions.csv',
    'items_file': 'items.csv',
    'similarity_matrix_file': 'similarity_matrix.npy',
    'item_index_file': 'item_index.pkl',
    'model_dir': 'models/',
    'data_dir': 'data/',
}

# region 随机种子 / region Random Seed
RANDOM_SEED = 42
