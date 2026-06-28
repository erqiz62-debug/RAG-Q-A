#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试向量数据库添加功能
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

# 创建向量数据库
vector_db = LocalVectorDatabase("./data/chroma_db", "medical_knowledge")

# 重置向量数据库
vector_db.reset()

# 只测试添加前10个chunks
test_chunks = data['chunks'][:10]
print(f"\n测试添加 {len(test_chunks)} 个chunks...")

try:
    vector_db.add_chunks(test_chunks, embedding_model=None)
    print("✅ 成功添加chunks")
    
    # 检查统计信息
    stats = vector_db.get_collection_stats()
    print(f"\n向量数据库统计:")
    print(f"  总chunks: {stats['total_chunks']}")
    print(f"  倒排索引大小: {stats['inverted_index_size']}")
    
except Exception as e:
    print(f"❌ 添加chunks失败: {e}")
    import traceback
    traceback.print_exc()
