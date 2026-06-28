#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试本地检索功能
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

print(f"总chunks数量: {len(data['chunks'])}")

# 创建向量数据库并添加chunks
vector_db = LocalVectorDatabase("./data/chroma_db", "medical_knowledge")

# 重置向量数据库
vector_db.reset()

# 添加所有chunks
print(f"\n添加所有chunks到向量数据库...")
vector_db.add_chunks(data['chunks'], embedding_model=None)

# 获取统计信息
stats = vector_db.get_collection_stats()
print(f"\n向量数据库统计:")
print(f"  总chunks: {stats['total_chunks']}")
print(f"  倒排索引大小: {stats['inverted_index_size']}")

# 测试检索
query = "高血压的症状有哪些？"
print(f"\n测试查询: {query}")

# 生成查询的embedding（使用hash向量）
query_embedding = vector_db._generate_hash_embedding(query)

# 混合检索（向量 + 关键词）
results = vector_db.hybrid_search(
    query_embedding=query_embedding,
    query=query,
    top_k=5,
    vector_weight=0.7,  # 70%向量检索
    keyword_weight=0.3  # 30%关键词检索
)

print(f"\n检索结果 ({len(results)} 个):")
for i, result in enumerate(results, 1):
    print(f"\n[{i}] 融合分数: {result['fused_score']:.4f}")
    print(f"    向量分数: {result['vector_score']:.4f}")
    print(f"    关键词分数: {result['keyword_score']:.4f}")
    print(f"    文档: {result['metadata'].get('_document_name', 'unknown')}")
    print(f"    页码: {result['metadata'].get('page', 'unknown')}")
    
    # 获取chunk内容
    chunk_content = vector_db.get_chunk_content(result['chunk_id'])
    if chunk_content:
        print(f"    内容: {chunk_content['content'][:100]}...")
    else:
        print(f"    内容: 无法获取")
