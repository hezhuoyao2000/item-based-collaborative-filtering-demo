# -*- coding: utf-8 -*-
"""
全局数据管理器
Global Data Manager
统一管理数据访问，确保数据一致性
Unified management of data access to ensure data consistency
"""

import os
from enhanced_data_processor import EnhancedDataProcessor
import config

class DataManager:
    """全局数据管理器 - 单例模式 / Global Data Manager - Singleton Pattern"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化数据管理器 / Initialize the data manager"""
        if self._initialized:
            return
            
        self.processor = None
        self.data_loaded = False
        self.data_generated = False  # 新增：标记数据是否已生成 / New: Flag to indicate if data has been generated
        self._initialized = True
    
    #region 获取数据处理器实例 / Get Data Processor Instance
    def get_processor(self):
        """获取数据处理器实例 / Get data processor instance"""
        if self.processor is None:
            self.processor = EnhancedDataProcessor()
        return self.processor
    #endregion
    #region 确保数据已加载 / Ensure Data is Loaded
    def ensure_data_loaded(self, force_regenerate=False):
        """确保数据已加载
        Ensure data is loaded
        Args:
            force_regenerate: 是否强制重新生成数据 / Whether to force data regeneration
        """
        if not self.data_loaded or force_regenerate:
            processor = self.get_processor()
            
            # 检查是否有保存的数据文件 / Check for saved data files
            data_files = [
                os.path.join(config.FILE_CONFIG['data_dir'], 'interactions.csv'),
                os.path.join(config.FILE_CONFIG['data_dir'], 'items.csv')
            ]
            
            if all(os.path.exists(f) for f in data_files) and not force_regenerate:
                print("加载保存的数据文件... / Loading saved data files...")
                processor.load_data(
                    interactions_file=data_files[0],
                    items_file=data_files[1]
                )
                processor.clean_data()
                processor.split_train_test()
                processor.create_enhanced_user_item_matrix('train')
            else:
                if force_regenerate:
                    print("强制重新生成数据... / Forcing data regeneration...")
                else:
                    print("未发现保存的数据文件，重新生成数据... / No saved data files found, regenerating data...")
                processor.generate_complete_dataset()
                self.data_generated = True
            
            self.data_loaded = True
        
        return self.get_processor()
    
    #region 获取训练数据 / Get Training Data
    def get_training_data(self):
        """获取训练数据 / Get training data"""
        processor = self.ensure_data_loaded()
        return processor.user_item_matrix, processor.item_index
    
    #endregion
    #region 获取测试数据 / Get Test Data
    def get_test_data(self):
        """获取测试数据 / Get test data"""
        processor = self.ensure_data_loaded()
        return processor.test_df
    
    #endregion
    #region 获取商品数据 / Get Item Data
    def get_items_data(self):
        """获取商品数据 / Get item data"""
        processor = self.ensure_data_loaded()
        return processor.items_df
    
    #endregion
    #region 获取交互数据 / Get Interaction Data
    def get_interactions_data(self):
        """获取交互数据 / Get interaction data"""
        processor = self.ensure_data_loaded()
        return processor.interactions_df
    
    #endregion
    #region 获取所有数据 / Get All Data
    def get_all_data(self):
        """获取所有数据 / Get all data"""
        processor = self.ensure_data_loaded()
        return {
            'interactions_df': processor.interactions_df,
            'items_df': processor.items_df,
            'train_df': processor.train_df,
            'test_df': processor.test_df,
            'user_item_matrix': processor.user_item_matrix,
            'user_index': processor.user_index,
            'item_index': processor.item_index
        }
    
    #endregion
    #region 重置数据管理器 / Reset Data Manager
    def reset(self):
        """重置数据管理器 / Reset data manager"""
        self.processor = None
        self.data_loaded = False
        self.data_generated = False
        print("数据管理器已重置 / Data manager has been reset")
    
    #endregion
    #region 检查数据是否已加载 / Check if Data is Loaded
    def is_data_loaded(self):
        """检查数据是否已加载 / Check if data is loaded"""
        return self.data_loaded
    
    #endregion
    #region 检查数据是否已生成 / Check if Data is Generated
    def is_data_generated(self):
        """检查数据是否已生成 / Check if data is generated"""
        return self.data_generated
    
    #endregion
    #region 检查数据文件是否存在 / Check if Data Files Exist
    def check_data_files_exist(self):
        """检查数据文件是否存在 / Check if data files exist"""
        data_files = [
            os.path.join(config.FILE_CONFIG['data_dir'], 'interactions.csv'),
            os.path.join(config.FILE_CONFIG['data_dir'], 'items.csv')
        ]
        return all(os.path.exists(f) for f in data_files)
    
    #endregion
    #region 获取数据摘要信息 / Get Data Summary Information
    def get_data_summary(self):
        """获取数据摘要信息 / Get data summary information"""
        if not self.data_loaded:
            return "数据未加载 / Data not loaded"
        
        processor = self.get_processor()
        summary = {
            'users': len(processor.interactions_df['user_id'].unique()) if processor.interactions_df is not None else 0,
            'items': len(processor.interactions_df['item_id'].unique()) if processor.interactions_df is not None else 0,
            'interactions': len(processor.interactions_df) if processor.interactions_df is not None else 0,
            'train_interactions': len(processor.train_df) if processor.train_df is not None else 0,
            'test_interactions': len(processor.test_df) if processor.test_df is not None else 0
        }
        return summary
    #endregion
# 全局数据管理器实例 / Global data manager instance
data_manager = DataManager()
