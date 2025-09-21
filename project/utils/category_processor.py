# -*- coding: utf-8 -*-
"""
分类数据处理模块
用于处理category.json文件，提取第一层和第二层分类数据创建映射
"""

import json
import os
from typing import List, Dict, Any

def process_category_data(input_file: str) -> Dict[str, Any]:
    """
    处理分类数据，提取第一层和第二层分类创建映射
    
    Args:
        input_file: 输入的category.json文件路径
    
    Returns:
        处理后的分类数据字典
    """
    print(f"正在处理分类数据: {input_file}")
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        category_data = json.load(f)
    
    print(f"原始数据包含 {len(category_data)} 个顶级分类")
    
    # 统计原始数据
    original_stats = get_category_statistics(category_data)
    print(f"原始统计: 第一层={original_stats['level1']}, 第二层={original_stats['level2']}")
    
    # 创建分类映射（只包含第一层和第二层）
    mapping = create_category_mapping(category_data)
    print(f"总共创建了 {len(mapping)} 个分类映射")
    
    # 保存分类映射到文件
    mapping_file = "category_mapping.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"分类映射已保存到: {mapping_file}")
    
    return category_data

def get_category_statistics(category_data: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    统计分类数据的层级分布
    
    Args:
        category_data: 分类数据
    
    Returns:
        包含各层级数量的字典
    """
    stats = {
        'level1': 0,
        'level2': 0
    }
    
    for category in category_data:
        stats['level1'] += 1
        
        if 'Children' in category and category['Children']:
            stats['level2'] += len(category['Children'])
    
    return stats

def create_category_mapping(category_data: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    创建分类ID到名称的映射（只包含第二层子分类）
    
    Args:
        category_data: 分类数据
    
    Returns:
        分类ID到名称的映射字典
    """
    mapping = {}
    
    for category in category_data:
        # 只提取第二层分类（子分类）
        if 'Children' in category and category['Children']:
            for child in category['Children']:
                mapping[child['Id']] = child['Name']
    
    return mapping

def analyze_category_structure(category_data: List[Dict[str, Any]]) -> None:
    """
    分析分类结构并打印详细信息
    
    Args:
        category_data: 分类数据
    """
    print("\n=== 分类结构分析 ===")
    
    for i, category in enumerate(category_data):
        print(f"\n第{i+1}个顶级分类:")
        print(f"  ID: {category['Id']}")
        print(f"  名称: {category['Name']}")
        print(f"  子分类数量: {len(category.get('Children', []))}")
        
        if category.get('Children'):
            for j, child in enumerate(category['Children'][:3]):  # 只显示前3个
                print(f"    {j+1}. {child['Name']} (ID: {child['Id']})")
            
            if len(category['Children']) > 3:
                print(f"    ... 还有 {len(category['Children']) - 3} 个子分类")

def main():
    """主函数"""
    # 文件路径
    input_file = "C:/Users/hezhuoyao/Desktop/work/recoomandation/project/category.json"
    
    if not os.path.exists(input_file):
        print(f"错误: 文件 {input_file} 不存在")
        return
    
    # 处理分类数据并创建映射
    category_data = process_category_data(input_file)
    
    # 分析分类结构
    analyze_category_structure(category_data)

if __name__ == "__main__":
    main()
