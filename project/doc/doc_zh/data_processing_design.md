# 数据处理模块设计文档

## 模块概述

数据处理模块是推荐系统的核心基础设施，负责数据的加载、清洗、转换和特征工程。该模块集成了真实分类数据，支持从原始数据到模型训练所需格式的完整处理流程。

## 核心组件

### EnhancedDataProcessor - 增强数据处理器

#### 核心功能
- **数据加载**: 支持CSV文件加载和实时数据生成
- **数据清洗**: 去重、无效值处理、冷启动过滤
- **数据划分**: 按时间划分训练集和测试集
- **矩阵构建**: 创建用户-物品交互矩阵
- **分类处理**: 集成真实分类体系元数据

## 数据处理流程

```
完整数据处理流程
1. 加载分类映射数据
   ↓
2. 加载/生成商品数据
   ↓
3. 加载/生成交互数据  
   ↓
4. 数据清洗和验证
   ↓
5. 训练集/测试集划分
   ↓
6. 用户-物品矩阵构建
   ↓
7. 统计信息生成
   ↓
8. 数据持久化保存
```

## 关键技术实现

### 1. 分类数据处理

#### 分类体系结构
```python
def _extract_categories_from_mapping(self):
    # 从分类映射中提取层级信息
    # 自动推断父分类关系
    # 构建完整的分类元数据
```

#### 分类权重计算
- 基于配置的热门分类列表
- 支持自定义分类权重
- 自动归一化处理

### 2. 数据清洗策略

#### 去重处理
```python
def clean_data(self):
    # 去除重复的用户-商品-时间记录
    # 过滤无效的用户ID和商品ID
    # 去除交互次数过少的用户和商品（冷启动处理）
```

#### 冷启动处理
- 用户至少2次交互才保留
- 商品至少2次交互才保留  
- 确保模型训练的有效性

### 3. 时间序列划分

#### 时间感知划分
```python
def split_train_test(self):
    # 按时间顺序排序交互数据
    # 前80%作为训练集，后20%作为测试集
    # 保持时间连续性，避免数据泄露
```

#### 时间范围统计
- 训练集时间范围显示
- 测试集时间范围显示
- 时间跨度分析

### 4. 稀疏矩阵构建

#### 交互矩阵构建
```python
def create_enhanced_user_item_matrix(self, data_type='train'):
    # 创建用户和商品的索引映射
    # 基于交互类型权重计算用户-物品权重
    # 构建CSR格式稀疏矩阵
```

#### 权重计算策略
- 点击: 权重1
- 加购: 权重3
- 收藏: 权重3
- 累加多次交互的权重

## 数据结构设计

### 1. 交互数据格式
```python
interactions_df: DataFrame
- user_id: 用户ID
- item_id: 商品ID  
- interaction_type: 交互类型（click/cart/favorite）
- category: 商品分类
- timestamp: 时间戳
```

### 2. 商品数据格式
```python
items_df: DataFrame  
- item_id: 商品ID
- category: 第二层分类名称
- price: 商品价格
- popularity_score: 流行度评分
- category_id: 分类ID
- parent_category: 第一层分类名称
- parent_category_id: 父分类ID
```

### 3. 矩阵数据结构
```python
user_item_matrix: csr_matrix
- 行: 用户索引
- 列: 商品索引
- 值: 交互权重总和

user_index: Dict[str, int]  # 用户ID到行索引的映射
item_index: Dict[str, int]  # 商品ID到列索引的映射
```

## 统计监控体系

### 1. 基础统计
```python
def get_statistics(self):
    # 用户数量、商品数量、交互数量
    # 交互类型分布统计
    # 用户交互次数分布（均值、中位数、极值）
    # 商品交互次数分布（均值、中位数、极值）
```

### 2. 分类统计
```python
def get_category_statistics(self):
    # 第二层分类分布（商品数量）
    # 第一层分类分布（商品数量）
    # 分类总数统计
```

### 3. 矩阵统计
```python
def get_matrix_statistics(self):
    # 矩阵形状和密度
    # 非零元素数量
    # 稀疏性分析
```

## 性能优化策略

### 1. 内存优化
- 使用稀疏矩阵存储大规模数据
- 分批处理避免内存溢出
- 及时释放临时变量

### 2. 计算优化
- 向量化操作替代循环
- 使用高效的数据结构
- 并行处理独立任务

### 3. IO优化
- 批量读写文件操作
- 使用二进制格式保存模型
- 缓存常用数据

## 扩展性设计

### 1. 数据源扩展
- 支持多种文件格式（CSV、JSON、Parquet）
- 支持数据库连接
- 支持实时数据流

### 2. 处理流程扩展
- 可插拔的数据清洗步骤
- 自定义的特征工程管道
- 灵活的数据划分策略

### 3. 分类体系扩展
- 支持多层级分类结构
- 动态加载分类映射
- 自定义分类权重

## 质量控制

### 1. 数据验证
- 数据类型检查
- 值范围验证
- 一致性检查

### 2. 异常处理
- 文件不存在处理
- 数据格式错误处理
- 内存不足处理

### 3. 日志监控
- 处理进度日志
- 错误警告日志
- 性能统计日志

## 使用示例

### 基本使用
```python
# 创建处理器实例
processor = EnhancedDataProcessor()

# 加载分类数据
processor.load_category_data('category_mapping.json')

# 生成完整数据集
results = processor.generate_complete_dataset()

# 或者加载现有数据
processor.load_data('interactions.csv', 'items.csv')
processor.clean_data()
processor.split_train_test()
matrix = processor.create_enhanced_user_item_matrix('train')
```

### 统计信息获取
```python
# 获取基础统计
stats = processor.get_statistics()

# 获取分类统计  
category_stats = processor.get_category_statistics()

# 获取用户交互历史
user_history = processor.get_user_interactions('u123', 'train')
```

## 最佳实践

1. **数据规模控制**: 根据内存容量调整数据规模
2. **定期清理**: 处理完成后及时清理中间数据
3. **备份机制**: 重要数据定期备份
4. **监控告警**: 设置处理时间阈值告警
5. **版本管理**: 数据处理流程版本化管理

该模块为推荐系统提供了稳定、高效、可扩展的数据处理能力，是系统可靠运行的重要保障。