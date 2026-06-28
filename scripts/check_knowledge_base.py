#!/usr/bin/env python3
"""
检查知识库内容和结构
"""

import json
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from precise_citation_medical_qa_system import PreciseCitationMedicalQASystem

def main():
    """检查知识库"""
    print("="*60)
    print("知识库内容检查")
    print("="*60)
    
    try:
        # 初始化系统
        knowledge_base_path = "../data/precise_citation_knowledge_base.json"
        system = PreciseCitationMedicalQASystem(knowledge_base_path)
        
        print(f"✓ 知识库加载成功")
        print(f"总知识块数: {len(system.knowledge_base)}")
        
        # 显示前5个知识块的详细信息
        print(f"\n前5个知识块详情:")
        print("-" * 40)
        
        for i, chunk in enumerate(system.knowledge_base[:5]):
            print(f"\n=== 知识块 {i+1} ===")
            print(f"ID: {chunk.get('id', 'N/A')}")
            print(f"内容长度: {len(chunk.get('content', ''))} 字符")
            print(f"内容预览: {chunk.get('content', '')[:150]}...")
            
            metadata = chunk.get('metadata', {})
            print(f"文档标题: {metadata.get('document_title', 'N/A')}")
            print(f"页码: {metadata.get('page_numbers', [])}")
            print(f"章节: {metadata.get('chapter', 'N/A')}")
            
        # 搜索包含"心脏病"的内容
        print(f"\n搜索包含'心脏病'的内容:")
        print("-" * 40)
        
        heart_disease_chunks = []
        for chunk in system.knowledge_base:
            content = chunk.get('content', '').lower()
            if '心脏病' in content:
                heart_disease_chunks.append(chunk)
        
        print(f"找到 {len(heart_disease_chunks)} 个包含'心脏病'的知识块")
        
        if heart_disease_chunks:
            for i, chunk in enumerate(heart_disease_chunks[:3]):
                print(f"\n--- 心脏病相关块 {i+1} ---")
                print(f"ID: {chunk.get('id', 'N/A')}")
                print(f"内容: {chunk.get('content', '')[:200]}...")
                metadata = chunk.get('metadata', {})
                print(f"来源: {metadata.get('document_title', 'N/A')} 第{metadata.get('page_numbers', [])}页")
        
        # 测试查询扩展
        print(f"\n测试查询扩展功能:")
        print("-" * 40)
        
        test_query = "心脏病是什么"
        print(f"原始查询: {test_query}")
        
        # 模拟核心概念提取
        core_concept = system._extract_core_concept(test_query)
        print(f"提取的核心概念: {core_concept}")
        
        # 模拟查询扩展
        synonyms = system._load_synonyms()
        if core_concept in synonyms:
            expanded_queries = [core_concept] + synonyms[core_concept]
        else:
            expanded_queries = [core_concept]
        print(f"扩展查询数量: {len(expanded_queries)}")
        print(f"前10个扩展查询: {expanded_queries[:10]}")
        
    except Exception as e:
        print(f"✗ 检查失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()