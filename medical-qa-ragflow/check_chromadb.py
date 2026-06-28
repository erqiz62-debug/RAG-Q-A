#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查ChromaDB状态
"""

import chromadb
import sys

try:
    print("检查ChromaDB状态...")
    print(f"ChromaDB版本: {chromadb.__version__}")
    
    # 连接到ChromaDB
    client = chromadb.PersistentClient(path='./data/chroma_db')
    print("✅ ChromaDB连接成功")
    
    # 获取集合
    try:
        collection = client.get_collection('medical_knowledge')
        print(f"✅ 集合 'medical_knowledge' 存在")
        count = collection.count()
        print(f"📊 集合中的文档数量: {count}")
        
        if count > 0:
            # 获取一些示例数据
            results = collection.get(limit=5)
            print(f"📝 示例数据:")
            for i, (doc_id, metadata) in enumerate(zip(results['ids'][:3], results['metadatas'][:3])):
                print(f"  [{i+1}] ID: {doc_id}")
                print(f"      文档: {metadata.get('_document_name', 'unknown')}")
                print(f"      页码: {metadata.get('page', 'unknown')}")
        else:
            print("⚠️  集合为空，没有添加任何文档")
            
    except Exception as e:
        print(f"❌ 获取集合失败: {e}")
        
except ImportError:
    print("❌ ChromaDB未安装")
    print("💡 安装命令: pip install chromadb")
    sys.exit(1)
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
