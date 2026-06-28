#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试查询优化效果
"""

from precise_citation_medical_qa_system import PreciseCitationMedicalQASystem

def test_query_optimization():
    """测试查询优化效果"""
    print("=" * 80)
    print("查询优化测试")
    print("=" * 80)
    
    # 初始化系统
    qa = PreciseCitationMedicalQASystem(
        knowledge_base_path="d:\\BS\\medical-qa-ragflow\\data\\precise_citation_knowledge_base.json"
    )
    
    # 测试两个等价查询
    queries = ["心脏病是什么", "什么是心脏病"]
    
    for i, query in enumerate(queries, 1):
        print(f"\n【测试查询 {i}】: {query}")
        print("-" * 60)
        
        # 处理查询
        result = qa.process_query_with_precise_citations(query)
        
        # 显示结果
        print(f"置信度: {result['confidence']:.3f}")
        print(f"引用数量: {len(result['precise_citations'])}")
        
        # 显示查询分析
        query_analysis = result.get('query_analysis', {})
        if query_analysis:
            print(f"查询类型: {query_analysis.get('query_type', 'unknown')}")
            print(f"核心概念: {query_analysis.get('core_concept', 'unknown')}")
            print(f"扩展查询数量: {query_analysis.get('expanded_queries_count', 0)}")
        
        # 显示回答
        print(f"回答预览: {result['answer'][:150]}...")
        
        # 显示精确溯源信息
        if result['precise_citations']:
            print("精确溯源信息:")
            for j, citation in enumerate(result['precise_citations'][:2], 1):
                print(f"  引用{j}: {citation['source_document']} - {citation['page_range']}")
        
        print("-" * 60)
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_query_optimization()