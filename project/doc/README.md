# 电商推荐系统 - 基于物品协同过滤

## 项目概述

这是一个基于物品协同过滤（Item-Based Collaborative Filtering）的电商推荐系统，集成了真实分类数据、用户偏好建模、混合推荐策略和全面的评估体系。系统支持从数据生成、模型训练到推荐生成和效果评估的完整流程。

## 核心特性

- **真实分类数据集成**: 使用真实的商品分类体系，支持多层级分类结构
- **智能用户偏好建模**: 基于用户兴趣强度生成个性化偏好数据
- **混合推荐策略**: 结合相似度推荐和分类偏好推荐，提升推荐质量
- **冷启动处理**: 针对新用户和低交互用户提供有效的冷启动解决方案
- **全面评估体系**: 支持传统准确性指标和兴趣把握指标的双重评估
- **可扩展架构**: 模块化设计，支持多种相似度算法和推荐策略

## 系统架构

```
推荐系统架构
├── 数据层 (Data Layer)
│   ├── 用户偏好生成器 (UserGenerator)
│   ├── 商品分配器 (ItemGenerator) 
│   └── 交互生成器 (InteractionGenerator)
├── 处理层 (Processing Layer)
│   └── 增强数据处理器 (EnhancedDataProcessor)
├── 模型层 (Model Layer)
│   └── 模型训练器 (ModelTrainer)
├── 推荐层 (Recommendation Layer)
│   └── 推荐生成器 (Recommender)
└── 应用层 (Application Layer)
    └── 主控制器 (Main)
```

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行示例

1. **生成完整数据集**
```bash
python main.py generate_data
```

2. **训练推荐模型**
```bash
python main.py train_model
```

3. **评估模型效果**
```bash
python main.py evaluate
```

4. **为用户生成推荐**
```bash
python main.py recommend --user_id u123
```

5. **查看系统状态**
```bash
python main.py status
```

## 配置说明

系统配置集中在 `config.py` 文件中，主要包括：

- **数据配置**: 用户数量、商品数量、交互数量等
- **模型配置**: 相似度算法、推荐数量等
- **文件配置**: 数据文件路径、模型文件路径等
- **分类配置**: 分类权重、热门分类设置等

## 模块详细说明

各模块的详细设计说明请参考 `doc/` 目录下的设计文档：

- [数据生成模块设计](data_generation_design.md)
- [数据处理模块设计](data_processing_design.md) 
- [模型训练模块设计](model_training_design.md)
- [推荐生成模块设计](recommendation_design.md)
- [系统架构设计](system_architecture.md)

## 性能指标

系统支持多种评估指标：

- **传统准确性指标**: Precision@K, Recall@K, F1-Score@K
- **兴趣把握指标**: 分类匹配率、兴趣对齐度、多样性得分
- **用户体验指标**: 推荐覆盖率、个性化程度、冷启动效果

## 扩展性

系统设计具有良好的扩展性：

- 支持多种相似度算法（余弦、皮尔逊、修正余弦）
- 可轻松添加新的推荐策略
- 支持自定义分类体系和用户偏好模型
- 模块化设计便于功能扩展和维护

## 技术支持

如有技术问题或建议，请联系项目维护团队。