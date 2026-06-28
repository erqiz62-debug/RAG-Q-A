#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试关键词检索功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from local_data_loader import LocalDataLoader
from local_vector_database import LocalVectorDatabase

# 创建数据加载器
loader = LocalDataLoader("D:\\Study\\py_program\\导出分块\\数据")

# 加载数据
data = loader.load_all_data()

# 创建向量数据库并添加chunks
vector_db = LocalVectorDatabase("./data/chroma_db", "medical_knowledge")

# 重置向量数据库
vector_db.reset()

# 添加所有chunks
print(f"添加所有chunks到向量数据库...")
vector_db.add_chunks(data['chunks'], embedding_model=None)

# 获取统计信息
stats = vector_db.get_collection_stats()
print(f"\n向量数据库统计:")
print(f"  总chunks: {stats['total_chunks']}")
print(f"  倒排索引大小: {stats['inverted_index_size']}")

# 测试关键词检索
query = "高血压的症状有哪些？"
print(f"\n测试查询: {query}")

# 关键词检索
results = vector_db.keyword_search(query, top_k=5)

print(f"\n关键词检索结果 ({len(results)} 个):")
for i, result in enumerate(results, 1):
    print(f"\n[{i}] 关键词分数: {result['keyword_score']:.4f}")
    print(f"    文档: {result['metadata'].get('_document_name', 'unknown')}")
    print(f"    页码: {result['metadata'].get('page', 'unknown')}")
    
    # 获取chunk内容
    chunk_content = vector_db.get_chunk_content(result['chunk_id'])
    if chunk_content:
        print(f"    内容: {chunk_content['content'][:100]}...")
    else:
        print(f"    内容: 无法获取")

# 测试单个关键词
print(f"\n测试单个关键词检索:")
test_keywords = ["高血压", "症状", "病理学", "细胞"]
for keyword in test_keywords:
    results = vector_db.keyword_search(keyword, top_k=3)
    print(f"\n关键词 '{keyword}': 找到 {len(results)} 个结果")
    if results:
        print(f"  第一个结果分数: {results[0]['keyword_score']:.4f}")
