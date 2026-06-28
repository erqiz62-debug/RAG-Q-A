#!/usr/bin/env python3
"""
测试不同检索模式效果的脚本
验证向量检索和关键词检索的对比效果
"""

import sys
import os
import json

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from precise_citation_medical_qa_system import PreciseCitationMedicalQASystem

def main():
    """主测试函数"""
    print("="*80)
    print("医学RAG系统检索模式对比测试")
    print("="*80)
    
    # 初始化系统
    try:
        knowledge_base_path = "../data/precise_citation_knowledge_base.json"
        system = PreciseCitationMedicalQASystem(knowledge_base_path)
        print("✓ 系统初始化成功")
    except Exception as e:
        print(f"✗ 系统初始化失败: {e}")
        return
    
    # 测试查询列表
    test_queries = [
        "什么是心脏病",
        "心脏病是什么", 
        "冠心病的症状有哪些",
        "高血压如何治疗",
        "心肌梗死的急救措施",
        "心电图检查的作用"
    ]
    
    print(f"\n准备测试 {len(test_queries)} 个查询...")
    
    # 对每个查询进行测试
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'-'*60}")
        print(f"查询 {i}: {query}")
        print(f"{'-'*60}")
        
        try:
            # 测试检索模式
            result = system.test_search_modes(query)
            
            # 显示详细对比
            print(f"\n📊 检索效果统计:")
            print(f"   向量检索结果数: {result['vector_count']}")
            print(f"   关键词检索结果数: {result['keyword_count']}")
            print(f"   共同结果数: {result['common_results']}")
            
            # 分析结果
            if result['vector_count'] == 0 and result['keyword_count'] == 0:
                print("   ⚠️  两种检索都未找到结果")
            elif result['vector_count'] > result['keyword_count']:
                print("   ✓ 向量检索效果更好")
            elif result['keyword_count'] > result['vector_count']:
                print("   ✓ 关键词检索效果更好")
            else:
                print("   ✓ 两种检索效果相当")
                
        except Exception as e:
            print(f"   ✗ 测试失败: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("测试完成！")
    print("="*80)
    
    # 总结分析
    print("\n📋 测试总结:")
    print("1. 向量检索：基于语义相似度，能够理解查询的深层含义")
    print("2. 关键词检索：基于精确匹配，速度快但缺乏语义理解")
    print("3. 混合检索：结合两者优势，提供最佳检索效果")
    print("\n💡 建议：对于医学知识库，建议使用混合检索模式")

if __name__ == "__main__":
    main()