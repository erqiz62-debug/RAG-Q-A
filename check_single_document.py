#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查单个文档的chunks.json文件
"""

import json
from pathlib import Path

# 选择一个文档文件夹
doc_dir = Path("D:\\Study\\py_program\\导出分块\\数据\\06885677_part_1.pdf_5510a540")

chunks_file = doc_dir / "chunks.json"
metadata_file = doc_dir / "metadata.json"

if chunks_file.exists():
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"文档: {doc_dir.name}")
    print(f"chunks数量: {len(chunks)}")
    
    # 检查前5个chunks
    print(f"\n前5个chunks:")
    for i, chunk in enumerate(chunks[:5], 1):
        print(f"  [{i}] chunk_id: {chunk.get('chunk_id', 'MISSING')}")
        print(f"      content长度: {len(chunk.get('content', ''))}")
        print(f"      chunk_index: {chunk.get('chunk_index', 'MISSING')}")
    
    # 检查chunk_id是否唯一
    chunk_ids = [chunk.get('chunk_id', '') for chunk in chunks]
    unique_ids = set(chunk_ids)
    
    print(f"\nchunk_id统计:")
    print(f"  总数: {len(chunk_ids)}")
    print(f"  唯一数: {len(unique_ids)}")
    print(f"  重复数: {len(chunk_ids) - len(unique_ids)}")
    
    # 找出重复的chunk_id
    from collections import Counter
    id_counts = Counter(chunk_ids)
    duplicates = {id: count for id, count in id_counts.items() if count > 1}
    
    if duplicates:
        print(f"\n重复的chunk_id（前5个）:")
        for i, (chunk_id, count) in enumerate(list(duplicates.items())[:5], 1):
            print(f"  [{i}] {chunk_id}: {count} 次")
    else:
        print("✅ 没有重复的chunk_id")
    
    # 检查chunk_index是否唯一
    chunk_indices = [chunk.get('chunk_index', -1) for chunk in chunks]
    unique_indices = set(chunk_indices)
    
    print(f"\nchunk_index统计:")
    print(f"  总数: {len(chunk_indices)}")
    print(f"  唯一数: {len(unique_indices)}")
    print(f"  重复数: {len(chunk_indices) - len(unique_indices)}")
    
    # 找出重复的chunk_index
    index_counts = Counter(chunk_indices)
    index_duplicates = {index: count for index, count in index_counts.items() if count > 1}
    
    if index_duplicates:
        print(f"\n重复的chunk_index（前5个）:")
        for i, (index, count) in enumerate(list(index_duplicates.items())[:5], 1):
            print(f"  [{i}] {index}: {count} 次")
    else:
        print("✅ 没有重复的chunk_index")

if metadata_file.exists():
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"\n文档元数据:")
    print(f"  document_id: {metadata.get('document_id', 'MISSING')}")
    print(f"  document_name: {metadata.get('document_name', 'MISSING')}")
    print(f"  dataset_id: {metadata.get('dataset_id', 'MISSING')}")
