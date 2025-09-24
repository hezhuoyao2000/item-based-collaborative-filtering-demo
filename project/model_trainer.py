# -*- coding: utf-8 -*-
"""
模型训练器模块
Model Trainer Module
使用训练集数据计算物品间的相似度，生成物品相似度矩阵
Uses training data to calculate item similarity and generate an item similarity matrix
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pickle
import os
import time
from functools import partial
from config import MODEL_CONFIG, FILE_CONFIG, RANDOM_SEED, SIMILARITY_CONFIG

class ModelTrainer:
    """模型训练器类 / Model Trainer Class"""
    
    def __init__(self):
        """初始化模型训练器 / Initialize the model trainer"""
        np.random.seed(RANDOM_SEED)
        self.config = MODEL_CONFIG
        self.file_config = FILE_CONFIG
        
        # 存储训练结果 / Store training results
        self.similarity_matrix = None
        self.item_index = None
        self.user_item_matrix = None
        
    #region 数据加载 / Data Loading
    #
    def load_training_data(self, user_item_matrix, item_index):
        """
        加载训练数据 / Load training data
        """
        print("正在加载训练数据... / Loading training data...")
        
        self.user_item_matrix = user_item_matrix
        self.item_index = item_index
        
        print(f"用户-物品矩阵形状: {self.user_item_matrix.shape} / User-item matrix shape: {self.user_item_matrix.shape}")
        print(f"商品数量: {len(self.item_index)} / Number of items: {len(self.item_index)}")
        
        return self.user_item_matrix, self.item_index
    #endregion

    #region 相似度计算 / Similarity Calculation
    def compute_item_similarity(self, similarity_metric='cosine'):
        """
        计算物品相似度矩阵
        Compute item similarity matrix
        相似度算法是相对固定的，主要变化在于：
        The similarity algorithm is relatively fixed, with main variations in:
        1. 数据预处理方式 / 1. Data preprocessing method
        2. 阈值参数设置 / 2. Threshold parameter settings
        3. 性能优化策略 / 3. Performance optimization strategies
        
        Args:
            similarity_metric: 相似度计算方法 / Similarity calculation method
        """
        print(f"正在计算物品相似度矩阵 (方法: {similarity_metric})... / Computing item similarity matrix (method: {similarity_metric})...")
        
        # 转置矩阵，得到物品-用户矩阵 / Transpose the matrix to get the item-user matrix
        item_user_matrix = self.user_item_matrix.T
        
        print(f"物品-用户矩阵形状: {item_user_matrix.shape} / Item-user matrix shape: {item_user_matrix.shape}")
        
        # 相似度计算方法映射 - 算法逻辑相对固定 / Similarity calculation method mapping - algorithm logic is relatively fixed
        similarity_methods = {
            'cosine': self._compute_cosine_similarity,
            'pearson': self._compute_pearson_similarity,
            'adjusted_cosine': self._compute_adjusted_cosine_similarity
        }
        
        if similarity_metric not in similarity_methods:
            raise ValueError(f"不支持的相似度计算方法: {similarity_metric} / Unsupported similarity calculation method: {similarity_metric}")
        
        # 执行相似度计算 - 核心算法逻辑不变 / Execute similarity calculation - core algorithm logic remains unchanged
        start_time = time.time()
        print(f"开始计算相似度矩阵，使用 {similarity_metric} 方法... / Starting to compute similarity matrix using {similarity_metric} method...")
        
        self.similarity_matrix = similarity_methods[similarity_metric](item_user_matrix)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"相似度矩阵计算完成，耗时: {elapsed_time:.2f} 秒 / Similarity matrix calculation complete, time taken: {elapsed_time:.2f} seconds")
        
        # 后处理 - 通用步骤 / Post-processing - common steps
        self._post_process_similarity_matrix()
        
        print(f"相似度矩阵形状: {self.similarity_matrix.shape} / Similarity matrix shape: {self.similarity_matrix.shape}")
        print(f"相似度矩阵密度: {np.count_nonzero(self.similarity_matrix) / self.similarity_matrix.size:.4f} / Similarity matrix density: {np.count_nonzero(self.similarity_matrix) / self.similarity_matrix.size:.4f}")
        
        return self.similarity_matrix
    
    def _post_process_similarity_matrix(self):
        """
        相似度矩阵后处理 - 通用步骤
        Similarity matrix post-processing - common steps
        这部分逻辑相对固定，适用于所有相似度方法
        This part of the logic is relatively fixed and applies to all similarity methods
        """
        common_config = SIMILARITY_CONFIG['common']
        
        # 将对角线设为0（物品与自己的相似度） / Set diagonal to 0 (similarity of an item with itself)
        if common_config['zero_diagonal']:
            np.fill_diagonal(self.similarity_matrix, 0)
        
        # 应用最小相似度阈值 / Apply minimum similarity threshold
        min_similarity = common_config['min_similarity']
        self.similarity_matrix[self.similarity_matrix < min_similarity] = 0
        
        # 确保矩阵对称性（如果需要） / Ensure matrix symmetry (if required)
        if common_config['symmetric'] and not np.allclose(self.similarity_matrix, self.similarity_matrix.T):
            print("警告: 相似度矩阵不对称，进行对称化处理 / Warning: Similarity matrix is not symmetric, performing symmetrization")
            self.similarity_matrix = (self.similarity_matrix + self.similarity_matrix.T) / 2
    #endregion

    #region 余弦相似度计算 / Cosine Similarity Calculation
    def _compute_cosine_similarity(self, item_user_matrix):
        """
        计算余弦相似度矩阵
        Compute cosine similarity matrix
        """
        print("计算余弦相似度... / Computing cosine similarity...")
        
        # 转换为密集矩阵进行余弦相似度计算
        # Convert to dense matrix for cosine similarity calculation
        # 注意：对于大数据集，这里可能需要优化内存使用
        # Note: For large datasets, memory usage may need to be optimized here
        dense_matrix = item_user_matrix.toarray()
        
        # 计算余弦相似度 - 核心算法逻辑固定 / Compute cosine similarity - core algorithm logic is fixed
        similarity_matrix = cosine_similarity(dense_matrix)
        
        return similarity_matrix
    #endregion

    #region 皮尔逊相关系数计算 / Pearson Correlation Coefficient Calculation
    def _compute_pearson_similarity(self, item_user_matrix):
        """
        计算皮尔逊相关系数相似度矩阵（优化版本）
        Compute Pearson correlation coefficient similarity matrix (optimized version)
        """
        print("计算皮尔逊相关系数（优化版本）... / Computing Pearson correlation coefficient (optimized version)...")
        
        # 转换为密集矩阵 / Convert to dense matrix
        dense_matrix = item_user_matrix.toarray()
        n_items = dense_matrix.shape[0]
        
        print(f"开始计算 {n_items} 个物品的相似度矩阵... / Starting to compute similarity matrix for {n_items} items...")
        
        # 使用numpy的corrcoef进行批量计算，比循环快很多 / Use numpy's corrcoef for batch calculation, much faster than looping
        try:
            # 计算所有物品对的皮尔逊相关系数 / Calculate Pearson correlation coefficient for all item pairs
            correlation_matrix = np.corrcoef(dense_matrix)
            
            # 处理NaN值（当标准差为0时） / Handle NaN values (when standard deviation is 0)
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
            
            # 将对角线设为0 / Set diagonal to 0
            np.fill_diagonal(correlation_matrix, 0)
            
            print("皮尔逊相关系数计算完成（使用向量化操作） / Pearson correlation coefficient calculation complete (using vectorized operations)")
            return correlation_matrix
            
        except MemoryError:
            print("内存不足，回退到分块计算... / Insufficient memory, falling back to chunked calculation...")
            return self._compute_pearson_similarity_chunked(dense_matrix)
    
    #region 分块计算皮尔逊相关系数 / Chunked calculation of Pearson correlation coefficient
    def _compute_pearson_similarity_chunked(self, dense_matrix):
        """
        分块计算皮尔逊相关系数（内存优化版本）
        Chunked calculation of Pearson correlation coefficient (memory optimized version)
        """
        from scipy.stats import pearsonr
        
        n_items = dense_matrix.shape[0]
        similarity_matrix = np.zeros((n_items, n_items))
        
        # 计算总的比较次数 / Calculate total number of comparisons
        total_comparisons = n_items * (n_items - 1) // 2
        completed = 0
        
        print(f"总共需要计算 {total_comparisons} 个物品对 / Total {total_comparisons} item pairs need to be calculated")
        
        # 使用更大的步长来减少进度显示频率 / Use a larger step size to reduce progress display frequency
        progress_step = max(1000, total_comparisons // 100)
        
        for i in range(n_items):
            for j in range(i+1, n_items):
                # 获取两个物品的评分向量 / Get rating vectors for two items
                item_i_ratings = dense_matrix[i]
                item_j_ratings = dense_matrix[j]
                
                # 找到两个物品都有评分的用户 / Find users who rated both items
                common_users = (item_i_ratings > 0) & (item_j_ratings > 0)
                
                # 使用配置参数 / Use configuration parameters
                min_common_users = SIMILARITY_CONFIG['pearson']['min_common_users']
                handle_nan = SIMILARITY_CONFIG['pearson']['handle_nan']
                
                if np.sum(common_users) < min_common_users:
                    similarity = 0
                else:
                    # 获取共同用户的评分 / Get ratings of common users
                    ratings_i = item_i_ratings[common_users]
                    ratings_j = item_j_ratings[common_users]
                    
                    # 检查是否有常量数组（所有值相同） / Check for constant arrays (all values are the same)
                    if np.std(ratings_i) == 0 or np.std(ratings_j) == 0:
                        similarity = 0
                    else:
                        # 计算皮尔逊相关系数 / Calculate Pearson correlation coefficient
                        correlation, _ = pearsonr(ratings_i, ratings_j)
                        if handle_nan and np.isnan(correlation):
                            similarity = 0
                        else:
                            similarity = correlation
                
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # 对称矩阵 / Symmetric matrix
                
                # 更新进度 / Update progress
                completed += 1
                if completed % progress_step == 0 or completed == total_comparisons:
                    progress = (completed / total_comparisons) * 100
                    print(f"进度: {completed}/{total_comparisons} ({progress:.1f}%) - 物品对 ({i}, {j}) / Progress: {completed}/{total_comparisons} ({progress:.1f}%) - Item pair ({i}, {j})")
        
        return similarity_matrix
    #endregion

    #region 修正余弦相似度计算 / Adjusted Cosine Similarity Calculation
    def _compute_adjusted_cosine_similarity(self, item_user_matrix):
        """
        计算修正余弦相似度矩阵
        Compute adjusted cosine similarity matrix
        """
        print("计算修正余弦相似度... / Computing adjusted cosine similarity...")
        
        # 转换为密集矩阵 / Convert to dense matrix
        dense_matrix = item_user_matrix.toarray()
        
        # 计算每个用户的平均评分 / Calculate average rating for each user
        user_means = np.zeros(dense_matrix.shape[1])
        for user_idx in range(dense_matrix.shape[1]):
            user_ratings = dense_matrix[:, user_idx]
            non_zero_ratings = user_ratings[user_ratings > 0]
            if len(non_zero_ratings) > 0:
                user_means[user_idx] = np.mean(non_zero_ratings)
        
        # 修正评分：减去用户平均评分 / Adjust ratings: subtract user average rating
        adjusted_matrix = dense_matrix.copy()
        for user_idx in range(dense_matrix.shape[1]):
            user_ratings = dense_matrix[:, user_idx]
            non_zero_mask = user_ratings > 0
            adjusted_matrix[non_zero_mask, user_idx] = user_ratings[non_zero_mask] - user_means[user_idx]
        
        # 计算修正后的余弦相似度 - 核心算法逻辑固定 / Compute adjusted cosine similarity - core algorithm logic is fixed
        similarity_matrix = cosine_similarity(adjusted_matrix)
        
        return similarity_matrix
    #endregion

    #region 相似物品查询 / Similar Item Query
    def get_item_similarities(self, item_id, top_k=10):
        """
        获取与指定物品最相似的top_k个物品
        Get top_k most similar items to the specified item
        """
        if item_id not in self.item_index:
            raise ValueError(f"商品ID {item_id} 不存在 / Item ID {item_id} does not exist")
        
        item_idx = self.item_index[item_id]
        similarities = self.similarity_matrix[item_idx]
        
        # 获取top_k个最相似的物品 / Get top_k most similar items
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 创建结果字典 / Create result dictionary
        similar_items = {}
        for idx in top_indices:
            if similarities[idx] > 0:  # 只返回相似度大于0的物品 / Only return items with similarity greater than 0
                item_id_similar = list(self.item_index.keys())[list(self.item_index.values()).index(idx)]
                similar_items[item_id_similar] = similarities[idx]
        
        return similar_items
    #endregion

    #region 统计信息 / Statistics
    def get_similarity_statistics(self):
        """
        获取相似度矩阵的统计信息
        Get statistics for the similarity matrix
        """
        print("\n=== 相似度矩阵统计信息 / Similarity Matrix Statistics ===")
        
        # 基本统计 / Basic statistics
        non_zero_similarities = self.similarity_matrix[self.similarity_matrix > 0]
        
        print(f"相似度矩阵形状: {self.similarity_matrix.shape} / Similarity matrix shape: {self.similarity_matrix.shape}")
        print(f"非零相似度数量: {len(non_zero_similarities)} / Number of non-zero similarities: {len(non_zero_similarities)}")
        print(f"相似度范围: [{self.similarity_matrix.min():.4f}, {self.similarity_matrix.max():.4f}] / Similarity range: [{self.similarity_matrix.min():.4f}, {self.similarity_matrix.max():.4f}]")
        print(f"平均相似度: {non_zero_similarities.mean():.4f} / Average similarity: {non_zero_similarities.mean():.4f}")
        print(f"相似度中位数: {np.median(non_zero_similarities):.4f} / Median similarity: {np.median(non_zero_similarities):.4f}")
        
        # 每个物品的平均相似物品数量 / Average number of similar items per item
        item_similarity_counts = np.count_nonzero(self.similarity_matrix, axis=1)
        print(f"平均每个物品的相似物品数量: {item_similarity_counts.mean():.2f} / Average number of similar items per item: {item_similarity_counts.mean():.2f}")
        print(f"最大相似物品数量: {item_similarity_counts.max()} / Maximum number of similar items: {item_similarity_counts.max()}")
        print(f"最小相似物品数量: {item_similarity_counts.min()} / Minimum number of similar items: {item_similarity_counts.min()}")
        
        # 相似度分布 / Similarity distribution
        similarity_ranges = [
            (0.9, 1.0, "0.9-1.0"),
            (0.8, 0.9, "0.8-0.9"),
            (0.7, 0.8, "0.7-0.8"),
            (0.6, 0.7, "0.6-0.7"),
            (0.5, 0.6, "0.5-0.6"),
            (0.1, 0.5, "0.1-0.5")
        ]
        
        print("\n相似度分布: / Similarity Distribution:")
        for low, high, label in similarity_ranges:
            count = np.sum((self.similarity_matrix >= low) & (self.similarity_matrix < high))
            print(f"{label}: {count}")
        
        return {
            'matrix_shape': self.similarity_matrix.shape,
            'non_zero_count': len(non_zero_similarities),
            'similarity_range': (self.similarity_matrix.min(), self.similarity_matrix.max()),
            'avg_similarity': non_zero_similarities.mean(),
            'avg_similar_items_per_item': item_similarity_counts.mean()
        }
    #endregion

    #region 模型保存 / Model Saving
    def save_model(self):
        """
        保存训练好的模型
        Save the trained model
        """
        print("正在保存模型... / Saving model...")
        
        # 创建模型目录 / Create model directory
        os.makedirs(self.file_config['model_dir'], exist_ok=True)
        
        # 保存相似度矩阵 / Save similarity matrix
        similarity_path = os.path.join(self.file_config['model_dir'], 
                                     self.file_config['similarity_matrix_file'])
        np.save(similarity_path, self.similarity_matrix)
        
        # 保存物品索引映射 / Save item index mapping
        index_path = os.path.join(self.file_config['model_dir'], 
                                self.file_config['item_index_file'])
        with open(index_path, 'wb') as f:
            pickle.dump(self.item_index, f)
        
        print(f"相似度矩阵已保存到: {similarity_path} / Similarity matrix saved to: {similarity_path}")
        print(f"物品索引已保存到: {index_path} / Item index saved to: {index_path}")
        
        return similarity_path, index_path
    #endregion

    #region 模型加载 / Model Loading
    def load_model(self):
        """
        加载已保存的模型
        Load the saved model
        """
        print("正在加载模型... / Loading model...")
        
        # 加载相似度矩阵 / Load similarity matrix
        similarity_path = os.path.join(self.file_config['model_dir'], 
                                     self.file_config['similarity_matrix_file'])
        self.similarity_matrix = np.load(similarity_path)
        
        # 加载物品索引 / Load item index
        index_path = os.path.join(self.file_config['model_dir'], 
                                self.file_config['item_index_file'])
        with open(index_path, 'rb') as f:
            self.item_index = pickle.load(f)
        
        print(f"相似度矩阵已加载: {self.similarity_matrix.shape} / Similarity matrix loaded: {self.similarity_matrix.shape}")
        print(f"物品索引已加载: {len(self.item_index)} 个物品 / Item index loaded: {len(self.item_index)} items")
        
        return self.similarity_matrix, self.item_index
    #endregion

    #region 完整训练流程 / Complete Training Workflow
    def train(self, user_item_matrix, item_index):
        """
        执行完整的模型训练流程
        Execute the complete model training workflow
        """
        print("开始模型训练... / Starting model training...")
        
        # 1. 加载训练数据 / 1. Load training data
        self.load_training_data(user_item_matrix, item_index)
        
        # 2. 计算物品相似度 / 2. Compute item similarity
        similarity_metric = self.config['similarity_metric']
        self.compute_item_similarity(similarity_metric)
        
        # 3. 显示统计信息 / 3. Display statistics
        self.get_similarity_statistics()
        
        # 4. 保存模型 / 4. Save model
        self.save_model()
        
        print("模型训练完成! / Model training complete!")
        
        return {
            'similarity_matrix': self.similarity_matrix,
            'item_index': self.item_index
        }
    #endregion

    #region 模型验证 / Model Validation
    def validate_model(self, test_items=None):
        """
        验证模型质量（简单的验证）
        Validate model quality (simple validation)
        """
        print("正在验证模型... / Validating model...")
        
        if test_items is None:
            # 随机选择一些物品进行验证 / Randomly select some items for validation
            test_items = np.random.choice(list(self.item_index.keys()), 
                                        size=min(5, len(self.item_index)), 
                                        replace=False)
        
        print(f"验证物品: {test_items} / Validating items: {test_items}")
        
        for item_id in test_items:
            similar_items = self.get_item_similarities(item_id, top_k=5)
            print(f"\n物品 {item_id} 的最相似物品: / \nMost similar items for item {item_id}:")
            for similar_item, similarity in similar_items.items():
                print(f"  {similar_item}: {similarity:.4f}")
        
        return True
    #endregion
