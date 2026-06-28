#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分批构建知识库
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

# 分批处理
batch_size = 100
total_chunks = len(data['chunks'])

for i in range(0, total_chunks, batch_size):
    batch = data['chunks'][i:i + batch_size]
    print(f"\n处理批次 {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}: {len(batch)} 个chunks")
    
    try:
        vector_db.add_chunks(batch, embedding_model=None)
        print(f"✅ 批次 {i//batch_size + 1} 完成")
    except Exception as e:
        print(f"❌ 批次 {i//batch_size + 1} 失败: {e}")
        import traceback
        traceback.print_exc()
        break

# 获取最终统计信息
stats = vector_db.get_collection_stats()
print(f"\n📊 最终统计:")
print(f"  总chunks: {stats['total_chunks']}")
print(f"  倒排索引大小: {stats['inverted_index_size']}")

# 验证ChromaDB状态
import chromadb
client = chromadb.PersistentClient(path='./data/chroma_db')
collection = client.get_collection('medical_knowledge')
count = collection.count()
print(f"  ChromaDB集合中的文档总数: {count}")
