#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查文档加载情况
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

# 按文档统计chunks
from collections import defaultdict
doc_chunks = defaultdict(list)
for chunk in data['chunks']:
    doc_id = chunk.get('_document_id', '')
    chunk_id = chunk.get('chunk_id', '')
    doc_chunks[doc_id].append(chunk_id)

print(f"\n文档数量: {len(doc_chunks)}")

# 检查每个文档的chunk_id是否唯一
duplicates_by_doc = {}
for doc_id, chunk_ids in doc_chunks.items():
    unique_ids = set(chunk_ids)
    if len(chunk_ids) != len(unique_ids):
        duplicates_by_doc[doc_id] = len(chunk_ids) - len(unique_ids)

if duplicates_by_doc:
    print(f"\n有重复chunk_id的文档（前10个）:")
    for i, (doc_id, duplicate_count) in enumerate(list(duplicates_by_doc.items())[:10], 1):
        print(f"  [{i}] {doc_id}: {duplicate_count} 个重复")
else:
    print("✅ 所有文档的chunk_id都是唯一的")

# 检查是否有重复的文档
doc_ids = [chunk.get('_document_id', '') for chunk in data['chunks']]
unique_doc_ids = set(doc_ids)
print(f"\n总文档ID数量: {len(unique_doc_ids)}")
print(f"文档ID出现次数: {len(doc_ids)}")

# 检查文档ID是否重复
from collections import Counter
doc_id_counts = Counter(doc_ids)
doc_duplicates = {doc_id: count for doc_id, count in doc_id_counts.items() if count > 1}

if doc_duplicates:
    print(f"\n重复的文档ID（前10个）:")
    for i, (doc_id, count) in enumerate(list(doc_duplicates.items())[:10], 1):
        print(f"  [{i}] {doc_id}: {count} 次")
else:
    print("✅ 没有重复的文档ID")
