# -*- coding: utf-8 -*-
"""
推荐生成器模块
基于物品协同过滤的推荐逻辑实现
"""

import numpy as np
import pandas as pd
import os
from model_trainer import ModelTrainer
from enhanced_data_processor import EnhancedDataProcessor
from config import MODEL_CONFIG, FILE_CONFIG

#endregion

#region 推荐器类定义
class Recommender:
    """推荐生成器类"""
    
    def __init__(self):
        """初始化推荐器"""
        self.config = MODEL_CONFIG
        self.file_config = FILE_CONFIG
        
        # 存储模型数据
        self.similarity_matrix = None
        self.item_index = None
        self.index_to_item = None
        self.data_processor = None
        
        # 加载模型
        self._load_model()
    
    #region 模型加载
    def _load_model(self):
        """
        加载训练好的模型 / Load trained model
        """
        print("正在加载推荐模型...")
        
        try:
            # 使用ModelTrainer加载模型
            trainer = ModelTrainer()
            self.similarity_matrix, self.item_index = trainer.load_model()
            
            # 创建反向索引映射
            self.index_to_item = {idx: item_id for item_id, idx in self.item_index.items()}
            
            print(f"模型加载成功: {len(self.item_index)} 个物品")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("请先运行模型训练: python main.py train_model")
            raise e
    #endregion

    #region 核心推荐功能
    def get_user_history(self, user_id, data_processor=None):
        """
        获取用户的历史交互记录
        """
        if data_processor is None:
            if self.data_processor is None:
                self.data_processor = EnhancedDataProcessor()
                # 尝试加载保存的数据
                data_files = [
                    os.path.join(self.file_config['data_dir'], 'enhanced_interactions.csv'),
                    os.path.join(self.file_config['data_dir'], 'enhanced_items.csv')
                ]
                
                if all(os.path.exists(f) for f in data_files):
                    self.data_processor.load_data(
                        interactions_file=data_files[0],
                        items_file=data_files[1]
                    )
                else:
                    self.data_processor.generate_complete_dataset()
                
                self.data_processor.clean_data()
                self.data_processor.split_train_test()
            
            data_processor = self.data_processor
        
        # 获取用户在训练集中的交互历史
        if hasattr(data_processor, 'get_user_interactions'):
            user_interactions = data_processor.get_user_interactions(user_id, data_type='train')
        else:
            # 兼容性处理：直接从训练集获取用户交互
            user_interactions = data_processor.train_df[data_processor.train_df['user_id'] == user_id]
        
        return user_interactions
    #endregion

    #region 评分计算
    def get_user_item_scores(self, user_id, data_processor=None):
        """
        计算用户对所有物品的评分
        基于用户历史交互的物品的相似物品，考虑交互强度
        """
        # 获取用户历史
        user_history = self.get_user_history(user_id, data_processor)
        
        if len(user_history) == 0:
            print(f"用户 {user_id} 没有历史交互记录")
            return {}
        
        # 定义交互类型权重
        interaction_weights = {
            'click': 1.0,
            'cart': 3.0,
            'favorite': 5.0
        }
        
        # 计算每个交互物品的权重
        item_weights = {}
        for _, interaction in user_history.iterrows():
            item_id = interaction['item_id']
            interaction_type = interaction['interaction_type']
            weight = interaction_weights.get(interaction_type, 1.0)
            
            if item_id not in item_weights:
                item_weights[item_id] = 0.0
            item_weights[item_id] += weight
        
        # 获取用户交互过的物品ID
        interacted_items = set(user_history['item_id'].unique())
        
        # 计算每个物品的推荐分数
        item_scores = {}
        
        for item_id in self.item_index.keys():
            if item_id in interacted_items:
                continue  # 跳过用户已经交互过的物品
            
            weighted_score = 0.0
            total_weight = 0.0
            item_idx = self.item_index[item_id]
            
            # 基于用户交互过的物品计算推荐分数，考虑交互强度
            for interacted_item in interacted_items:
                if interacted_item in self.item_index:
                    interacted_idx = self.item_index[interacted_item]
                    similarity = self.similarity_matrix[item_idx, interacted_idx]
                    item_weight = item_weights[interacted_item]
                    
                    weighted_score += similarity * item_weight
                    total_weight += item_weight
            
            # 加权平均分数
            if total_weight > 0:
                score = weighted_score / total_weight
                
                # 评分放大，提高区分度
                score = score * 2.0  # 放大因子
                
                if score > 0:
                    item_scores[item_id] = score
        
        return item_scores
    #endregion

    #region 推荐生成和策略
    def get_recommendations(self, user_id, top_n=10, data_processor=None):
        """
        为用户生成Top-N推荐（混合策略）
        
        Args:
            user_id: 用户ID
            top_n: 推荐商品数量
            data_processor: 数据处理器实例
        
        Returns:
            list: 推荐商品列表，格式为 [(item_id, score), ...]
        """
        # 检查是否为冷启动用户
        if self._is_cold_start_user(user_id, data_processor):
            return self._handle_cold_start_recommendations(user_id, top_n, data_processor)
        
        # 混合推荐策略
        recommendations = self._generate_hybrid_recommendations(user_id, top_n, data_processor)
        
        return recommendations
    
    # region 生成混合推荐  / region Generate hybrid recommendations
    def _generate_hybrid_recommendations(self, user_id, top_n, data_processor):
        """
        生成混合推荐（相似度 + 分类偏好）
        """
        # 相似度推荐 (70%)
        similarity_count = int(top_n * 0.7)
        similarity_recs = self._get_similarity_recommendations(user_id, similarity_count, data_processor)
        
        # 分类偏好推荐 (30%)
        category_count = top_n - similarity_count
        category_recs = self._get_category_preference_recommendations(user_id, category_count, data_processor)
        
        # 合并推荐
        all_recommendations = similarity_recs + category_recs
        
        # 去重和重新排序
        unique_recommendations = self._deduplicate_recommendations(all_recommendations)
        
        return unique_recommendations[:top_n]
    #endregion

    #region 基于相似度的推荐  / region Recommendation based on similarity
    def _get_similarity_recommendations(self, user_id, top_n, data_processor):
        """
        基于相似度的推荐
        """
        # 获取用户物品评分
        item_scores = self.get_user_item_scores(user_id, data_processor)
        
        if not item_scores:
            return []
        
        # 按分数排序，获取top_n个推荐
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_n]
    #endregion
    #region 基于分类偏好的推荐  / region Recommendation based on category preference
    def _get_category_preference_recommendations(self, user_id, top_n, data_processor):
        """
        基于分类偏好的推荐，考虑交互强度
        """
        if data_processor is None:
            return []
        
        # 获取用户分类偏好
        user_categories = self._get_user_category_preferences(user_id, data_processor)
        
        if not user_categories:
            return []
        
        # 获取用户历史交互物品
        user_history = self.get_user_history(user_id, data_processor)
        interacted_items = set(user_history['item_id'].unique())
        
        # 从偏好分类中推荐物品
        category_recommendations = []
        
        for category, weight in user_categories.items():
            # 获取该分类的物品
            category_items = self._get_items_by_category(category, data_processor)
            
            if not category_items:
                continue
            
            # 计算物品推荐分数（基于分类权重和物品相似度）
            for item_id in category_items:
                if item_id in self.item_index and item_id not in interacted_items:
                    # 基础分类权重分数
                    category_score = weight
                    
                    # 结合相似度信息
                    if item_id in self.item_index:
                        item_idx = self.item_index[item_id]
                        max_similarity = 0.0
                        
                        # 计算与用户历史物品的最大相似度
                        for interacted_item in interacted_items:
                            if interacted_item in self.item_index:
                                interacted_idx = self.item_index[interacted_item]
                                similarity = self.similarity_matrix[item_idx, interacted_idx]
                                max_similarity = max(max_similarity, similarity)
                        
                        # 综合分数：分类权重 + 相似度
                        final_score = category_score * 0.7 + max_similarity * 0.3
                        category_recommendations.append((item_id, final_score))
        
        # 按分数排序
        category_recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return category_recommendations[:top_n]
    
    #region 获取用户分类偏好  / region Get user category preferences
    def _get_user_category_preferences(self, user_id, data_processor):
        """
        获取用户分类偏好，考虑交互强度
        """
        # 获取用户历史交互
        user_history = self.get_user_history(user_id, data_processor)
        
        if len(user_history) == 0:
            return {}
        
        # 定义交互类型权重
        interaction_weights = {
            'click': 1.0,
            'cart': 3.0,
            'favorite': 5.0
        }
        
        # 获取用户交互的商品分类
        if hasattr(data_processor, 'items_df') and data_processor.items_df is not None:
            # 计算每个分类的加权交互次数
            category_weights = {}
            total_weighted_interactions = 0.0
            
            for _, interaction in user_history.iterrows():
                item_id = interaction['item_id']
                interaction_type = interaction['interaction_type']
                weight = interaction_weights.get(interaction_type, 1.0)
                
                # 获取商品分类
                item_category = data_processor.items_df[
                    data_processor.items_df['item_id'] == item_id
                ]['category'].iloc[0] if len(data_processor.items_df[
                    data_processor.items_df['item_id'] == item_id
                ]) > 0 else None
                
                if item_category:
                    if item_category not in category_weights:
                        category_weights[item_category] = 0.0
                    category_weights[item_category] += weight
                    total_weighted_interactions += weight
            
            # 归一化分类权重
            if total_weighted_interactions > 0:
                for category in category_weights:
                    category_weights[category] = category_weights[category] / total_weighted_interactions
            
            return category_weights
        return {}
    #endregion
    #region 获取指定分类的物品  / region Get items by category
    def _get_items_by_category(self, category, data_processor):
        """
        获取指定分类的物品
        """
        if hasattr(data_processor, 'items_df') and data_processor.items_df is not None:
            category_items = data_processor.items_df[
                data_processor.items_df['category'] == category
            ]['item_id'].tolist()
            return category_items
        
        return []
    #endregion
    #region 去重推荐结果  / region Deduplicate recommendations
    def _deduplicate_recommendations(self, recommendations):
        """
        去重推荐结果
        """
        seen_items = set()
        unique_recommendations = []
        
        for item_id, score in recommendations:
            if item_id not in seen_items:
                seen_items.add(item_id)
                unique_recommendations.append((item_id, score))
        
        return unique_recommendations
    #endregion
    #region 判断是否为冷启动用户  / region Check if user is a cold start user
    def _is_cold_start_user(self, user_id, data_processor):
        """
        判断是否为冷启动用户
        """
        user_history = self.get_user_history(user_id, data_processor)
        return len(user_history) < 3  # 交互次数少于3次认为是冷启动
    #endregion
    #region 处理冷启动用户推荐  / region Handle cold start user recommendations
    def _handle_cold_start_recommendations(self, user_id, top_n, data_processor):
        """
        处理冷启动用户推荐
        """
        if data_processor is None:
            return []
        
        # 策略1: 基于全局热门商品
        popular_items = self._get_global_popular_items(data_processor, top_n)
        
        # 策略2: 基于分类热门商品（如果有用户信息）
        if hasattr(data_processor, 'items_df') and data_processor.items_df is not None:
            category_popular = self._get_category_popular_items(data_processor, top_n)
            popular_items.extend(category_popular)
        
        # 去重和排序
        unique_items = self._deduplicate_recommendations(popular_items)
        
        return unique_items[:top_n]
    #endregion
    #region 获取全局热门商品  / region Get global popular items
    def _get_global_popular_items(self, data_processor, top_n):
        """
        获取全局热门商品
        """
        if hasattr(data_processor, 'interactions_df') and data_processor.interactions_df is not None:
            # 统计商品交互次数
            item_counts = data_processor.interactions_df['item_id'].value_counts()
            popular_items = [(item_id, count) for item_id, count in item_counts.head(top_n).items()]
            return popular_items
        
        return []
    #endregion
    #endregion
    #region 获取分类热门商品  / region Get category popular items
    def _get_category_popular_items(self, data_processor, top_n):
        """
        获取分类热门商品
        """
        if not hasattr(data_processor, 'items_df') or data_processor.items_df is None:
            return []
        
        # 获取主要分类
        main_categories = data_processor.items_df['category'].value_counts().head(5).index
        
        category_items = []
        for category in main_categories:
            category_items_df = data_processor.items_df[data_processor.items_df['category'] == category]
            # 随机选择一些商品作为热门商品
            sample_items = category_items_df.sample(min(3, len(category_items_df)))['item_id'].tolist()
            for item_id in sample_items:
                category_items.append((item_id, 1.0))  # 简化分数
        
        return category_items
    #endregion
    #endregion
    #region 物品相似推荐
    def get_item_recommendations(self, item_id, top_n=10):
        """
        基于物品相似度获取推荐物品
        
        Args:
            item_id: 商品ID
            top_n: 推荐商品数量
        
        Returns:
            list: 推荐商品列表，格式为 [(item_id, similarity), ...]
        """
        if item_id not in self.item_index:
            print(f"商品 {item_id} 不存在")
            return []
        
        item_idx = self.item_index[item_id]
        similarities = self.similarity_matrix[item_idx]
        
        # 获取最相似的物品
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        recommendations = []
        for idx in top_indices:
            if similarities[idx] > 0:
                similar_item = self.index_to_item[idx]
                recommendations.append((similar_item, similarities[idx]))
        
        return recommendations
    #endregion
    #endregion
    #region 推荐解释
    def get_recommendation_explanations(self, user_id, recommendations, data_processor=None):
        """
        为推荐结果提供解释
        """
        if data_processor is None:
            if self.data_processor is None:
                self.data_processor = EnhancedDataProcessor()
                # 尝试加载保存的数据
                data_files = [
                    os.path.join(self.file_config['data_dir'], 'enhanced_interactions.csv'),
                    os.path.join(self.file_config['data_dir'], 'enhanced_items.csv')
                ]
                
                if all(os.path.exists(f) for f in data_files):
                    self.data_processor.load_data(
                        interactions_file=data_files[0],
                        items_file=data_files[1]
                    )
                else:
                    self.data_processor.generate_complete_dataset()
                
                self.data_processor.clean_data()
                self.data_processor.split_train_test()
            data_processor = self.data_processor
        
        # 获取用户历史
        user_history = self.get_user_history(user_id, data_processor)
        interacted_items = set(user_history['item_id'].unique())
        
        explanations = []
        
        for item_id, score in recommendations:
            # 找到最相似的交互物品
            item_idx = self.item_index[item_id]
            max_similarity = 0
            most_similar_item = None
            
            for interacted_item in interacted_items:
                if interacted_item in self.item_index:
                    interacted_idx = self.item_index[interacted_item]
                    similarity = self.similarity_matrix[item_idx, interacted_idx]
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_item = interacted_item
            
            explanation = {
                'recommended_item': item_id,
                'score': score,
                'most_similar_item': most_similar_item,
                'similarity': max_similarity,
                'explanation': f"推荐此商品是因为您之前对商品 {most_similar_item} 感兴趣，相似度为 {max_similarity:.3f}"
            }
            explanations.append(explanation)
        
        return explanations
    #endregion
    #endregion
    #region 用户画像
    def get_user_profile(self, user_id, data_processor=None):
        """
        获取用户画像信息
        """
        if data_processor is None:
            if self.data_processor is None:
                self.data_processor = EnhancedDataProcessor()
                # 尝试加载保存的数据
                data_files = [
                    os.path.join(self.file_config['data_dir'], 'enhanced_interactions.csv'),
                    os.path.join(self.file_config['data_dir'], 'enhanced_items.csv')
                ]
                
                if all(os.path.exists(f) for f in data_files):
                    self.data_processor.load_data(
                        interactions_file=data_files[0],
                        items_file=data_files[1]
                    )
                else:
                    self.data_processor.generate_complete_dataset()
                
                self.data_processor.clean_data()
                self.data_processor.split_train_test()
            data_processor = self.data_processor
        
        # 获取用户历史
        user_history = self.get_user_history(user_id, data_processor)
        
        if len(user_history) == 0:
            return {
                'user_id': user_id,
                'interaction_count': 0,
                'favorite_categories': [],
                'interaction_types': {}
            }
        
        # 统计交互类型
        interaction_types = user_history['interaction_type'].value_counts().to_dict()
        
        # 获取商品类别偏好（如果有商品数据）
        favorite_categories = []
        category_weights = {}
        if hasattr(data_processor, 'items_df') and data_processor.items_df is not None:
            # 获取用户交互的商品类别
            user_items = user_history['item_id'].unique()
            item_categories = data_processor.items_df[
                data_processor.items_df['item_id'].isin(user_items)
            ]['category'].value_counts()
            favorite_categories = item_categories.head(3).index.tolist()
            
            # 计算分类权重
            total_interactions = len(user_history)
            for category, count in item_categories.items():
                category_weights[category] = count / total_interactions
        
        # 分析用户兴趣特征
        interest_profile = self._analyze_user_interest(user_id, data_processor)
        
        profile = {
            'user_id': user_id,
            'interaction_count': len(user_history),
            'favorite_categories': favorite_categories,
            'category_weights': category_weights,
            'interaction_types': interaction_types,
            'interest_profile': interest_profile,
            'is_cold_start': len(user_history) < 3,
            'first_interaction': user_history['timestamp'].min(),
            'last_interaction': user_history['timestamp'].max()
        }
        
        return profile
    
    def _analyze_user_interest(self, user_id, data_processor):
        """
        分析用户兴趣特征
        """
        user_history = self.get_user_history(user_id, data_processor)
        
        if len(user_history) == 0:
            return {}
        
        interest_profile = {
            'interaction_diversity': 0.0,  # 交互多样性
            'category_diversity': 0.0,     # 分类多样性
            'interaction_intensity': 0.0,  # 交互强度
            'preference_stability': 0.0,   # 偏好稳定性
            'exploration_tendency': 0.0    # 探索倾向
        }
        
        # 计算交互多样性（基于交互类型）
        interaction_types = user_history['interaction_type'].value_counts()
        if len(interaction_types) > 1:
            # 使用香农熵计算多样性
            total = interaction_types.sum()
            entropy = 0
            for count in interaction_types:
                p = count / total
                if p > 0:
                    entropy -= p * np.log2(p)
            interest_profile['interaction_diversity'] = entropy / np.log2(len(interaction_types))
        
        # 计算分类多样性
        if hasattr(data_processor, 'items_df') and data_processor.items_df is not None:
            user_items = user_history['item_id'].unique()
            user_categories = data_processor.items_df[
                data_processor.items_df['item_id'].isin(user_items)
            ]['category'].value_counts()
            
            if len(user_categories) > 1:
                total = user_categories.sum()
                entropy = 0
                for count in user_categories:
                    p = count / total
                    if p > 0:
                        entropy -= p * np.log2(p)
                interest_profile['category_diversity'] = entropy / np.log2(len(user_categories))
        
        # 计算交互强度（平均每天交互次数）
        if len(user_history) > 1:
            # 由于timestamp是整数，直接计算差值
            time_span = user_history['timestamp'].max() - user_history['timestamp'].min()
            if time_span > 0:
                interest_profile['interaction_intensity'] = len(user_history) / time_span
        
        return interest_profile
    #endregion
    #endregion
    #region 批量推荐
    def batch_recommend(self, user_ids, top_n=10, data_processor=None):
        """
        批量推荐
        """
        print(f"正在为 {len(user_ids)} 个用户批量生成推荐...")
        
        batch_results = {}
        
        for user_id in user_ids:
            try:
                recommendations = self.get_recommendations(user_id, top_n, data_processor)
                batch_results[user_id] = recommendations
            except Exception as e:
                print(f"为用户 {user_id} 生成推荐失败: {e}")
                batch_results[user_id] = []
        
        print(f"批量推荐完成，成功处理 {len([r for r in batch_results.values() if r])} 个用户")
        
        return batch_results
    #endregion
    #endregion
    #region 推荐统计
    def get_recommendation_stats(self, recommendations):
        """
        获取推荐结果的统计信息
        """
        if not recommendations:
            return {}
        
        scores = [score for _, score in recommendations]
        
        stats = {
            'recommendation_count': len(recommendations),
            'avg_score': np.mean(scores),
            'max_score': np.max(scores),
            'min_score': np.min(scores),
            'score_std': np.std(scores)
        }
        
        return stats
    #endregion
    #region 评估推荐质量  / region Evaluate recommendation quality
    def evaluate_recommendation_quality(self, user_id, recommendations, data_processor=None):
        """
        评估推荐质量（聚焦兴趣把握精准度）
        """
        if data_processor is None:
            return {}
        
        # 获取用户兴趣画像
        user_profile = self.get_user_profile(user_id, data_processor)
        
        # 获取推荐商品的分类信息
        recommended_items = [item_id for item_id, _ in recommendations]
        
        if not recommended_items or not hasattr(data_processor, 'items_df'):
            return {}
        
        # 分析推荐商品的分类分布
        recommended_categories = data_processor.items_df[
            data_processor.items_df['item_id'].isin(recommended_items)
        ]['category'].value_counts()
        
        # 计算推荐质量指标
        quality_metrics = {
            'category_match_rate': 0.0,      # 分类匹配率
            'diversity_score': 0.0,          # 多样性得分
            'interest_alignment': 0.0,       # 兴趣对齐度
            'cold_start_effectiveness': 0.0  # 冷启动效果
        }
        
        # 1. 分类匹配率
        if 'category_weights' in user_profile and user_profile['category_weights']:
            user_categories = set(user_profile['category_weights'].keys())
            recommended_cat_set = set(recommended_categories.index)
            
            if user_categories:
                match_count = len(user_categories & recommended_cat_set)
                quality_metrics['category_match_rate'] = match_count / len(user_categories)
        
        # 2. 多样性得分（基于推荐分类的熵）
        if len(recommended_categories) > 1:
            total = recommended_categories.sum()
            entropy = 0
            for count in recommended_categories:
                p = count / total
                if p > 0:
                    entropy -= p * np.log2(p)
            quality_metrics['diversity_score'] = entropy / np.log2(len(recommended_categories))
        
        # 3. 兴趣对齐度（推荐分类与用户偏好的相似性）
        if 'category_weights' in user_profile and user_profile['category_weights']:
            user_weights = user_profile['category_weights']
            rec_weights = (recommended_categories / recommended_categories.sum()).to_dict()
            
            # 计算加权相似度
            all_categories = set(user_weights.keys()) | set(rec_weights.keys())
            if all_categories:
                user_vec = [user_weights.get(cat, 0) for cat in all_categories]
                rec_vec = [rec_weights.get(cat, 0) for cat in all_categories]
                
                # 余弦相似度
                dot_product = sum(a * b for a, b in zip(user_vec, rec_vec))
                norm_user = sum(a * a for a in user_vec) ** 0.5
                norm_rec = sum(b * b for b in rec_vec) ** 0.5
                
                if norm_user > 0 and norm_rec > 0:
                    quality_metrics['interest_alignment'] = dot_product / (norm_user * norm_rec)
        
        # 4. 冷启动效果（如果是冷启动用户）
        if user_profile.get('is_cold_start', False):
            # 基于推荐商品的全局受欢迎程度
            if hasattr(data_processor, 'interactions_df'):
                global_popularity = data_processor.interactions_df['item_id'].value_counts()
                rec_popularity = global_popularity[global_popularity.index.isin(recommended_items)]
                
                if len(rec_popularity) > 0:
                    avg_popularity = rec_popularity.mean()
                    max_popularity = global_popularity.max()
                    quality_metrics['cold_start_effectiveness'] = avg_popularity / max_popularity
        
        return quality_metrics
    #endregion
    #region 综合评估和指标计算
    def comprehensive_evaluate(self, test_users=None, max_users=100, data_processor=None):
        """
        综合评估推荐效果（整合原evaluator功能）
        """
        if data_processor is None:
            if self.data_processor is None:
                self.data_processor = EnhancedDataProcessor()
                # 尝试加载保存的数据
                data_files = [
                    os.path.join(self.file_config['data_dir'], 'enhanced_interactions.csv'),
                    os.path.join(self.file_config['data_dir'], 'enhanced_items.csv')
                ]
                
                if all(os.path.exists(f) for f in data_files):
                    self.data_processor.load_data(
                        interactions_file=data_files[0],
                        items_file=data_files[1]
                    )
                else:
                    self.data_processor.generate_complete_dataset()
                
                self.data_processor.clean_data()
                self.data_processor.split_train_test()
            data_processor = self.data_processor
        
        # 获取测试用户
        if test_users is None:
            test_users = self._get_test_users(data_processor)
        
        if max_users is not None:
            test_users = test_users[:max_users]
        
        print(f"开始综合评估，测试用户数量: {len(test_users)}")
        
        # 评估结果
        evaluation_results = {
            'accuracy_metrics': {},      # 传统准确性指标
            'interest_metrics': {},      # 兴趣把握指标
            'user_experience': {},       # 用户体验指标
            'cold_start_analysis': {}    # 冷启动分析
        }
        
        # 1. 传统准确性评估
        print("计算传统准确性指标...")
        evaluation_results['accuracy_metrics'] = self._evaluate_accuracy(test_users, data_processor)
        
        # 2. 兴趣把握精准度评估
        print("计算兴趣把握指标...")
        evaluation_results['interest_metrics'] = self._evaluate_interest_accuracy(test_users, data_processor)
        
        # 3. 用户体验评估
        print("计算用户体验指标...")
        evaluation_results['user_experience'] = self._evaluate_user_experience(test_users, data_processor)
        
        # 4. 冷启动分析
        print("分析冷启动效果...")
        evaluation_results['cold_start_analysis'] = self._analyze_cold_start(test_users, data_processor)
        
        # 打印评估结果
        self._print_evaluation_results(evaluation_results)
        
        return evaluation_results
    #endregion
    #region 获取测试用户  / region Get test users
    def _get_test_users(self, data_processor):
        """
        获取测试用户列表
        """
        # 获取在测试集中有交互的用户
        test_users = data_processor.test_df['user_id'].unique()
        
        # 过滤掉在训练集中没有交互的用户（冷启动问题）
        train_users = set(data_processor.train_df['user_id'].unique())
        valid_test_users = [user for user in test_users if user in train_users]
        
        print(f"测试用户数量: {len(valid_test_users)}")
        return valid_test_users
    #endregion
    #region 评估传统准确性指标  / region Evaluate traditional accuracy metrics
    def _evaluate_accuracy(self, test_users, data_processor):
        """
        评估传统准确性指标
        """
        k_values = [5, 10, 20]
        all_precisions = {k: [] for k in k_values}
        all_recalls = {k: [] for k in k_values}
        all_f1_scores = {k: [] for k in k_values}
        
        successful_evaluations = 0
        total_users = len(test_users)
        progress_interval = max(1, total_users // 10)  # 10%进度间隔
        
        for i, user_id in enumerate(test_users):
            try:
                # 获取推荐结果
                recommendations = self.get_recommendations(user_id, top_n=max(k_values), data_processor=data_processor)
                recommended_items = [item_id for item_id, _ in recommendations]
                
                # 获取真实交互物品
                user_test_interactions = data_processor.get_user_interactions(user_id, data_type='test')
                relevant_items = set(user_test_interactions['item_id'].unique())
                
                if len(relevant_items) == 0:
                    continue
                
                # 计算各种K值下的指标
                for k in k_values:
                    precision = self._calculate_precision_at_k(recommended_items, relevant_items, k)
                    recall = self._calculate_recall_at_k(recommended_items, relevant_items, k)
                    f1_score = self._calculate_f1_score_at_k(precision, recall)
                    
                    all_precisions[k].append(precision)
                    all_recalls[k].append(recall)
                    all_f1_scores[k].append(f1_score)
                
                successful_evaluations += 1
                
                # 进度报告：每10%显示一次
                if (i + 1) % progress_interval == 0 or (i + 1) == total_users:
                    progress_percent = ((i + 1) / total_users) * 100
                    print(f"评估进度: {i + 1}/{total_users} ({progress_percent:.1f}%)")
                
            except Exception as e:
                print(f"评估用户 {user_id} 失败: {e}")
                continue
        
        # 计算平均指标
        accuracy_metrics = {}
        for k in k_values:
            if all_precisions[k]:
                accuracy_metrics[f'avg_precision@{k}'] = np.mean(all_precisions[k])
                accuracy_metrics[f'avg_recall@{k}'] = np.mean(all_recalls[k])
                accuracy_metrics[f'avg_f1_score@{k}'] = np.mean(all_f1_scores[k])
                accuracy_metrics[f'std_precision@{k}'] = np.std(all_precisions[k])
                accuracy_metrics[f'std_recall@{k}'] = np.std(all_recalls[k])
                accuracy_metrics[f'std_f1_score@{k}'] = np.std(all_f1_scores[k])
        
        accuracy_metrics['successful_evaluations'] = successful_evaluations
        return accuracy_metrics
    #endregion
    #region 评估兴趣把握精准度  / region Evaluate interest accuracy
    def _evaluate_interest_accuracy(self, test_users, data_processor):
        """
        评估兴趣把握精准度
        """
        interest_metrics = {
            'category_match_rates': [],
            'interest_alignments': [],
            'diversity_scores': [],
            'cold_start_effectiveness': []
        }
        
        total_users = len(test_users)
        progress_interval = max(1, total_users // 10)  # 10%进度间隔
        
        for i, user_id in enumerate(test_users):
            try:
                # 获取推荐结果
                recommendations = self.get_recommendations(user_id, top_n=10, data_processor=data_processor)
                
                # 评估推荐质量
                quality_metrics = self.evaluate_recommendation_quality(user_id, recommendations, data_processor)
                
                if quality_metrics:
                    interest_metrics['category_match_rates'].append(quality_metrics.get('category_match_rate', 0))
                    interest_metrics['interest_alignments'].append(quality_metrics.get('interest_alignment', 0))
                    interest_metrics['diversity_scores'].append(quality_metrics.get('diversity_score', 0))
                    interest_metrics['cold_start_effectiveness'].append(quality_metrics.get('cold_start_effectiveness', 0))
                
                # 进度报告：每10%显示一次
                if (i + 1) % progress_interval == 0 or (i + 1) == total_users:
                    progress_percent = ((i + 1) / total_users) * 100
                    print(f"兴趣指标评估进度: {i + 1}/{total_users} ({progress_percent:.1f}%)")
                
            except Exception as e:
                print(f"评估用户 {user_id} 兴趣指标失败: {e}")
                continue
        
        # 计算平均指标
        result = {}
        for metric, values in interest_metrics.items():
            if values:
                result[f'avg_{metric}'] = np.mean(values)
                result[f'std_{metric}'] = np.std(values)
        
        return result
    #endregion
    #region 评估用户体验指标  / region Evaluate user experience metrics
    def _evaluate_user_experience(self, test_users, data_processor):
        """
        评估用户体验指标
        """
        user_experience = {
            'recommendation_coverage': 0.0,    # 推荐覆盖率
            'response_time': 0.0,              # 响应时间
            'explanation_quality': 0.0,        # 解释质量
            'personalization_degree': 0.0      # 个性化程度
        }
        
        # 计算推荐覆盖率
        all_recommended_items = set()
        total_users = len(test_users)
        progress_interval = max(1, total_users // 10)  # 10%进度间隔
        
        for i, user_id in enumerate(test_users):
            try:
                recommendations = self.get_recommendations(user_id, top_n=10, data_processor=data_processor)
                recommended_items = [item_id for item_id, _ in recommendations]
                all_recommended_items.update(recommended_items)
                
                # 进度报告：每10%显示一次
                if (i + 1) % progress_interval == 0 or (i + 1) == total_users:
                    progress_percent = ((i + 1) / total_users) * 100
                    print(f"用户体验评估进度: {i + 1}/{total_users} ({progress_percent:.1f}%)")
                    
            except:
                continue
        
        if hasattr(data_processor, 'items_df'):
            total_items = len(data_processor.items_df)
            user_experience['recommendation_coverage'] = len(all_recommended_items) / total_items
        
        # 计算个性化程度（用户间推荐差异）
        if len(test_users) > 1:
            user_recommendations = {}
            for i, user_id in enumerate(test_users):
                try:
                    recommendations = self.get_recommendations(user_id, top_n=5, data_processor=data_processor)
                    recommended_items = set([item_id for item_id, _ in recommendations])
                    user_recommendations[user_id] = recommended_items
                    
                    # 进度报告：每10%显示一次
                    if (i + 1) % progress_interval == 0 or (i + 1) == total_users:
                        progress_percent = ((i + 1) / total_users) * 100
                        print(f"个性化程度评估进度: {i + 1}/{total_users} ({progress_percent:.1f}%)")
                        
                except:
                    continue
            
            if len(user_recommendations) > 1:
                similarities = []
                user_list = list(user_recommendations.keys())
                for i in range(len(user_list)):
                    for j in range(i + 1, len(user_list)):
                        user1_items = user_recommendations[user_list[i]]
                        user2_items = user_recommendations[user_list[j]]
                        
                        if user1_items and user2_items:
                            similarity = len(user1_items & user2_items) / len(user1_items | user2_items)
                            similarities.append(similarity)
                
                if similarities:
                    user_experience['personalization_degree'] = 1 - np.mean(similarities)
        
        return user_experience
    #endregion
    #region 分析冷启动效果  / region Analyze cold start effect
    def _analyze_cold_start(self, test_users, data_processor):
        """
        分析冷启动效果
        """
        cold_start_analysis = {
            'cold_start_users': 0,
            'cold_start_success_rate': 0.0,
            'avg_cold_start_effectiveness': 0.0
        }
        
        cold_start_users = []
        cold_start_effectiveness = []
        
        total_users = len(test_users)
        progress_interval = max(1, total_users // 10)  # 10%进度间隔
        
        for i, user_id in enumerate(test_users):
            user_profile = self.get_user_profile(user_id, data_processor)
            if user_profile.get('is_cold_start', False):
                cold_start_users.append(user_id)
                
                # 评估冷启动效果
                recommendations = self.get_recommendations(user_id, top_n=10, data_processor=data_processor)
                quality_metrics = self.evaluate_recommendation_quality(user_id, recommendations, data_processor)
                
                if quality_metrics:
                    effectiveness = quality_metrics.get('cold_start_effectiveness', 0)
                    cold_start_effectiveness.append(effectiveness)
            
            # 进度报告：每10%显示一次
            if (i + 1) % progress_interval == 0 or (i + 1) == total_users:
                progress_percent = ((i + 1) / total_users) * 100
                print(f"冷启动分析进度: {i + 1}/{total_users} ({progress_percent:.1f}%)")
        
        cold_start_analysis['cold_start_users'] = len(cold_start_users)
        if cold_start_users:
            cold_start_analysis['cold_start_success_rate'] = len(cold_start_effectiveness) / len(cold_start_users)
            if cold_start_effectiveness:
                cold_start_analysis['avg_cold_start_effectiveness'] = np.mean(cold_start_effectiveness)
        
        return cold_start_analysis
    #endregion
    #region 计算Precision@K  / region Calculate Precision@K
    def _calculate_precision_at_k(self, recommended_items, relevant_items, k):
        """计算Precision@K"""
        if k == 0:
            return 0.0
        
        top_k_recommendations = recommended_items[:k]
        hits = len(set(top_k_recommendations) & relevant_items)
        return hits / k if k > 0 else 0.0
    #endregion
    #region 计算Recall@K  / region Calculate Recall@K
    def _calculate_recall_at_k(self, recommended_items, relevant_items, k):
        """计算Recall@K"""
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recommendations = recommended_items[:k]
        hits = len(set(top_k_recommendations) & relevant_items)
        return hits / len(relevant_items)
    #endregion
    #region 计算F1-Score@K  / region Calculate F1-Score@K
    def _calculate_f1_score_at_k(self, precision, recall):
        """计算F1-Score@K"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    #endregion
    #region 打印评估结果  / region Print evaluation results
    def _print_evaluation_results(self, evaluation_results):
        """
        打印评估结果
        """
        print("\n" + "="*60)
        print("推荐系统综合评估结果")
        print("="*60)
        
        # 传统准确性指标
        accuracy = evaluation_results['accuracy_metrics']
        if accuracy:
            print("\n【传统准确性指标】")
            for k in [5, 10, 20]:
                if f'avg_precision@{k}' in accuracy:
                    print(f"Top-{k}:")
                    print(f"  Precision: {accuracy[f'avg_precision@{k}']:.4f} ± {accuracy[f'std_precision@{k}']:.4f}")
                    print(f"  Recall:    {accuracy[f'avg_recall@{k}']:.4f} ± {accuracy[f'std_recall@{k}']:.4f}")
                    print(f"  F1-Score:  {accuracy[f'avg_f1_score@{k}']:.4f} ± {accuracy[f'std_f1_score@{k}']:.4f}")
        
        # 兴趣把握指标
        interest = evaluation_results['interest_metrics']
        if interest:
            print("\n【兴趣把握指标】")
            print(f"分类匹配率: {interest.get('avg_category_match_rates', 0):.4f} ± {interest.get('std_category_match_rates', 0):.4f}")
            print(f"兴趣对齐度: {interest.get('avg_interest_alignments', 0):.4f} ± {interest.get('std_interest_alignments', 0):.4f}")
            print(f"多样性得分: {interest.get('avg_diversity_scores', 0):.4f} ± {interest.get('std_diversity_scores', 0):.4f}")
        
        # 用户体验指标
        ux = evaluation_results['user_experience']
        if ux:
            print("\n【用户体验指标】")
            print(f"推荐覆盖率: {ux.get('recommendation_coverage', 0):.4f}")
            print(f"个性化程度: {ux.get('personalization_degree', 0):.4f}")
        
        # 冷启动分析
        cold_start = evaluation_results['cold_start_analysis']
        if cold_start:
            print("\n【冷启动分析】")
            print(f"冷启动用户数: {cold_start.get('cold_start_users', 0)}")
            print(f"冷启动成功率: {cold_start.get('cold_start_success_rate', 0):.4f}")
            print(f"平均冷启动效果: {cold_start.get('avg_cold_start_effectiveness', 0):.4f}")
        
        print("\n" + "="*60)
    #endregion
    #endregion
#endregion
