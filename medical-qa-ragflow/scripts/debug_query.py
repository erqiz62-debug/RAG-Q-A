#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试查询处理过程
"""

from precise_citation_medical_qa_system import PreciseCitationMedicalQASystem
import jieba

def debug_query_processing():
    """调试查询处理过程"""
    print("=" * 80)
    print("查询处理调试")
    print("=" * 80)
    
    # 初始化系统
    qa = PreciseCitationMedicalQASystem(
        knowledge_base_path="d:\\BS\\medical-qa-ragflow\\data\\precise_citation_knowledge_base.json"
    )
    
    # 测试查询
    query = "心脏病是什么"
    print(f"测试查询: {query}")
    print("-" * 60)
    
    # 1. 查询类型分类
    query_type = qa._classify_query_type(query)
    print(f"查询类型: {query_type}")
    
    # 2. 核心概念提取
    core_concept = qa._extract_core_concept(query)
    print(f"核心概念: {core_concept}")
    
    # 3. 医学术语检查
    is_medical = qa._is_medical_term(core_concept)
    print(f"是否为医学术语: {is_medical}")
    
    # 4. 同义词获取
    synonyms = qa._get_synonyms_in_query(query)
    print(f"同义词: {synonyms}")
    
    # 5. 查询扩展
    expanded_queries = qa.expand_query_for_broad_questions(query)
    print(f"扩展查询数量: {len(expanded_queries)}")
    print(f"前5个扩展查询: {expanded_queries[:5]}")
    
    # 6. 手动测试相关性计算
    print("\n" + "="*60)
    print("相关性计算调试")
    print("="*60)
    
    query_words = set(jieba.cut(query))
    query_clean = query.lower().strip()
    
    # 检查前3个知识块
    for i, chunk in enumerate(qa.knowledge_base[:3]):
        print(f"\n知识块 {i+1}:")
        print(f"章节: {chunk.get('chapter', 'N/A')}")
        print(f"小节: {chunk.get('section', 'N/A')}")
        print(f"内容前100字符: {chunk.get('content', '')[:100]}...")
        
        # 计算相关性
        score = qa._calculate_chunk_relevance(chunk, query, query_words, query_clean)
        print(f"相关性分数: {score:.6f}")
        
        # 显示详细计算过程
        content = chunk.get('content', '')
        content_lower = content.lower()
        
        # 直接匹配检查
        direct_match = query_clean in content_lower
        print(f"直接匹配: {direct_match}")
        
        # 部分匹配检查
        query_parts = query_clean.split()
        partial_matches = [part for part in query_parts if len(part) > 1 and part in content_lower]
        print(f"部分匹配: {partial_matches}")
        
        # 医学术语匹配
        medical_terms = chunk.get('medical_terms', [])
        term_matches = [term for term in medical_terms if term.lower() in query_clean or query_clean in term.lower()]
        print(f"医学术语匹配: {term_matches}")
        
        # 章节标题匹配
        chapter = chunk.get('chapter', '').lower()
        section = chunk.get('section', '').lower()
        title_matches = [word for word in query_parts if word in chapter or word in section]
        print(f"标题匹配: {title_matches}")
        
        print("-" * 40)
    
    # 7. 执行完整检索
    print("\n" + "="*60)
    print("完整检索结果")
    print("="*60)
    
    candidates = qa.intelligent_retrieve(query, top_k=5)
    print(f"检索到的候选结果数量: {len(candidates)}")
    
    for i, candidate in enumerate(candidates):
        chunk = candidate['chunk']
        print(f"\n候选结果 {i+1}:")
        print(f"相关性分数: {candidate['relevance_score']:.6f}")
        print(f"章节: {chunk.get('chapter', 'N/A')}")
        print(f"内容预览: {chunk.get('content', '')[:150]}...")

if __name__ == "__main__":
    debug_query_processing()