#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查chunks数据结构
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from local_data_loader import LocalDataLoader

# 创建数据加载器
loader = LocalDataLoader("D:\\Study\\py_program\\导出分块\\数据")

# 加载数据
data = loader.load_all_data()

print(f"总chunks数量: {len(data['chunks'])}")

# 检查前5个chunks的结构
print("\n前5个chunks的结构:")
for i, chunk in enumerate(data['chunks'][:5], 1):
    print(f"\n[{i}] Chunk:")
    print(f"  chunk_id: {chunk.get('chunk_id', 'MISSING')}")
    print(f"  content长度: {len(chunk.get('content', ''))}")
    print(f"  metadata keys: {list(chunk.get('metadata', {}).keys())}")
    print(f"  _document_name: {chunk.get('_document_name', 'MISSING')}")
    print(f"  embedding: {'存在' if chunk.get('embedding') else '不存在'}")
    
    # 检查metadata
    metadata = chunk.get('metadata', {})
    print(f"  metadata内容:")
    for key, value in metadata.items():
        print(f"    {key}: {value}")

# 统计没有chunk_id的chunks
no_id_count = sum(1 for c in data['chunks'] if not c.get('chunk_id'))
print(f"\n没有chunk_id的chunks数量: {no_id_count}")
