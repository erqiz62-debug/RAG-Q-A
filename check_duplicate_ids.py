#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查chunk_id重复问题
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

# 检查chunk_id唯一性
chunk_ids = [chunk.get('chunk_id', '') for chunk in data['chunks']]
unique_ids = set(chunk_ids)

print(f"唯一chunk_id数量: {len(unique_ids)}")
print(f"重复chunk_id数量: {len(chunk_ids) - len(unique_ids)}")

# 找出重复的chunk_id
from collections import Counter
id_counts = Counter(chunk_ids)
duplicates = {id: count for id, count in id_counts.items() if count > 1}

if duplicates:
    print(f"\n重复的chunk_id（前10个）:")
    for i, (chunk_id, count) in enumerate(list(duplicates.items())[:10], 1):
        print(f"  [{i}] {chunk_id}: {count} 次")
else:
    print("✅ 没有重复的chunk_id")
