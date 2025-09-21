# 数据生成模块设计文档

## 模块概述

数据生成模块负责创建模拟的电商推荐系统数据，包括用户偏好、商品信息和用户-商品交互数据。该模块基于真实分类体系，生成符合实际业务场景的模拟数据。

## 核心组件

### 1. UserGenerator - 用户偏好生成器

#### 核心功能
- 生成用户偏好类型（单一分类、多分类、探索型）
- 基于分类受欢迎程度分配用户兴趣强度
- 提供用户交互概率计算接口

#### 设计原理
- **偏好类型分布**: 30%单一分类用户、50%多分类用户、20%探索型用户
- **兴趣强度**: 高兴趣(30%)、中等兴趣(50%)、低兴趣(20%)
- **分类权重**: 基于配置文件的分类受欢迎程度进行加权随机选择

#### 关键算法
```python
def generate_category_interests(self, user_id, preference_type, categories):
    # 基于偏好类型生成分类兴趣
    if preference_type == 'single_category':
        # 只选择一个高兴趣分类
    elif preference_type == 'multi_category':
        # 选择2-5个分类，兴趣强度不同
    else:  # explorer
        # 对所有分类都有兴趣，强度基于受欢迎程度
```

### 2. ItemGenerator - 商品分配生成器

#### 核心功能
- 将商品分配到真实分类体系中
- 生成商品属性（价格、流行度等）
- 维护分类到商品的映射关系

#### 设计原理
- **分类权重分配**: 热门分类获得更多商品资源
- **属性生成**: 价格和流行度基于分类特性进行调整
- **数量控制**: 每个分类至少有5个商品，最多20个商品

#### 关键特性
- 支持真实分类映射
- 基于分类特性的属性调整
- 商品数量智能分配

### 3. InteractionGenerator - 交互生成器

#### 核心功能
- 基于用户兴趣生成用户-商品交互
- 支持多种交互类型（点击、加购、收藏）
- 避免重复交互，模拟真实用户行为

#### 设计原理
- **兴趣驱动**: 用户对偏好分类的商品有更高交互概率
- **探索行为**: 20%的概率进行随机探索
- **交互类型**: 点击(70%)、加购(20%)、收藏(10%)

#### 交互权重
- 点击: 权重1
- 加购: 权重3  
- 收藏: 权重3

## 数据流程

```
数据生成流程
1. 加载分类映射数据
2. 生成用户偏好数据
   ↓
3. 分配商品到分类
   ↓  
4. 生成用户-商品交互
   ↓
5. 数据清洗和预处理
   ↓
6. 保存到文件系统
```

## 配置参数

### 用户偏好配置
```python
user_preference: {
    'preference_types': {
        'single_category': 0.3,      # 30%单一分类用户
        'multi_category': 0.5,       # 50%多分类用户
        'explorer': 0.2              # 20%探索型用户
    },
    'max_categories_per_user': 5,    # 每用户最多5个感兴趣分类
    'min_categories_per_user': 1,    # 每用户最少1个感兴趣分类
}
```

### 分类兴趣配置
```python
category_interest: {
    'category_popularity': {
        'Baby & Toddler Food': 0.15,  # 婴儿食品最受欢迎
        'Fresh Foods & Bakery': 0.12,
        'Health & Body': 0.10,
        # ... 其他分类
    },
    'interest_based_probability': {
        'high_interest': 0.8,        # 高兴趣分类交互概率80%
        'medium_interest': 0.4,      # 中等兴趣40%
        'low_interest': 0.1          # 低兴趣10%
    }
}
```

### 商品分配配置
```python
enhanced_processor: {
    'popular_categories': [           # 热门分类列表
        'Baby & Toddler Food',
        'Health & Body', 
        'Household & Cleaning',
        'Pantry',
        'Fresh Foods & Bakery',
        'Hot & Cold Drinks'
    ],
    'category_weights': {
        'popular': 3.0,              # 热门分类权重
        'food_baby': 2.0,            # 食品和婴儿用品权重
        'default': 1.0               # 默认权重
    },
    'price_range': (10, 500),        # 价格范围
    'items_per_category': {
        'min': 5,                    # 每个分类最少5个商品
        'max': 20,                   # 每个分类最多20个商品
        'popular_multiplier': 2      # 热门分类商品数加倍
    }
}
```

## 数据质量保证

### 1. 数据清洗
- 去除重复的用户-商品-时间交互记录
- 过滤无效的用户ID和商品ID
- 去除交互次数过少的用户和商品（冷启动处理）

### 2. 数据划分
- 按时间顺序划分训练集和测试集
- 默认训练集比例80%，测试集20%
- 确保时间连续性，避免数据泄露

### 3. 统计验证
- 用户交互次数分布验证
- 商品交互次数分布验证  
- 分类分布统计验证
- 交互类型分布验证

## 扩展性设计

### 支持自定义分类体系
- 通过修改 `category_mapping.json` 文件
- 支持任意层级的分类结构
- 自动提取第二层分类用于推荐

### 灵活的参数配置
- 所有参数通过配置文件管理
- 支持动态调整数据规模
- 易于进行A/B测试

### 数据导出格式
- CSV格式便于其他系统使用
- JSON格式保存元数据信息
- 完整的统计信息报告

## 性能考虑

### 内存优化
- 使用生成器模式逐步生成数据
- 分批处理大规模数据
- 稀疏矩阵存储用户-物品交互

### 计算效率
- 向量化操作替代循环
- 基于权重的随机采样算法
- 并行化处理用户数据生成

## 使用示例

```python
# 生成完整数据集
processor = EnhancedDataProcessor()
results = processor.generate_complete_dataset()

# 获取统计信息
stats = processor.get_statistics()
category_stats = processor.get_category_statistics()
```

该模块为推荐系统提供了高质量、符合真实业务场景的模拟数据，为后续的模型训练和推荐生成奠定了坚实基础。