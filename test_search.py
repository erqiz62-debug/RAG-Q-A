#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试医学问答系统检索功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from vectorization_engine import MedicalVectorEngine

def test_search_functionality():
    """测试检索功能"""
    print("=== 测试医学知识检索功能 ===")
    
    # 初始化向量引擎
    engine = MedicalVectorEngine()
    
    # 测试查询
    test_queries = [
        "心脏病的病因和病理",
        "冠心病的诊断方法", 
        "心力衰竭的治疗",
        "高血压的临床表现"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        print("-" * 50)
        
        results = engine.search_similar_chunks(query, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                chunk = result["chunk"]
                similarity = result["similarity"]
                print(f"{i}. 相似度: {similarity:.4f}")
                print(f"   内容: {chunk['content'][:100]}...")
                print(f"   来源: {chunk.get('metadata', {}).get('document_title', 'N/A')}")
                print(f"   章节: {chunk.get('metadata', {}).get('chapter', 'N/A')}")
                print()
        else:
            print("未找到相关结果")

if __name__ == "__main__":
    test_search_functionality()