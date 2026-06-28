#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试核心概念提取修复效果
"""

import sys
sys.path.append('.')

from precise_citation_medical_qa_system import PreciseCitationMedicalQASystem

def test_core_concept_extraction():
    """测试核心概念提取功能"""
    print("=== 测试核心概念提取修复效果 ===")
    
    # 初始化系统
    try:
        qa_system = PreciseCitationMedicalQASystem('d:\\BS\\medical-qa-ragflow\\data\\precise_citation_knowledge_base.json')
        print("✓ 系统初始化成功")
    except Exception as e:
        print(f"✗ 系统初始化失败: {e}")
        return
    
    # 测试查询
    test_queries = [
        '心脏病是什么',
        '什么是心脏病',
        '心脏病是什么意思',
        '什么是高血压',
        '糖尿病是什么'
    ]
    
    print("\n=== 核心概念提取测试 ===")
    for query in test_queries:
        print(f"\n查询: \"{query}\"")
        core_concept = qa_system._extract_core_concept(query)
        print(f"  提取的核心概念: \"{core_concept}\"")
        
        # 测试概念关联术语
        related_terms = qa_system._get_concept_related_terms(core_concept)
        print(f"  相关术语: {related_terms}")
    
    print("\n=== 查询类型分类测试 ===")
    for query in test_queries:
        query_type = qa_system._classify_query_type(query)
        print(f"查询: \"{query}\" -> 类型: {query_type}")

if __name__ == "__main__":
    test_core_concept_extraction()