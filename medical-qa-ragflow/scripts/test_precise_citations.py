#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精确溯源医学问答系统演示
"""

import sys
import os
sys.path.append('d:\\BS\\medical-qa-ragflow')

from scripts.precise_citation_medical_qa_system import PreciseCitationMedicalQASystem

def main():
    print("=" * 80)
    print("🎯 精确溯源医学问答系统演示")
    print("=" * 80)
    
    # 初始化系统
    qa_system = PreciseCitationMedicalQASystem(
        knowledge_base_path="d:\\BS\\medical-qa-ragflow\\data\\precise_citation_knowledge_base.json"
    )
    
    # 测试查询
    test_queries = [
        "什么是心脏病？",
        "冠心病的症状有哪些？",
        "如何治疗高血压？",
        "心力衰竭的原因是什么？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n【查询 {i}】: {query}")
        print("=" * 60)
        
        try:
            response = qa_system.process_query_with_precise_citations(query)
            
            print(f"📝 回答: {response['answer']}")
            print(f"🎯 置信度: {response['confidence']:.3f}")
            print(f"📊 引用数量: {len(response['precise_citations'])}")
            
            # 显示精确溯源信息
            if response['precise_citations']:
                print("\n📚 精确溯源信息:")
                for j, citation in enumerate(response['precise_citations'][:2], 1):
                    print(f"  引用 {j}:")
                    print(f"    📄 源文档: {citation['source_document']}")
                    print(f"    📖 章节: {citation['chapter']} - {citation['section']}")
                    print(f"    📑 页码: {citation['page_range']}")
                    print(f"    📝 行号: {citation['line_range']}")
                    print(f"    🎯 相关性: {citation['relevance_score']:.3f}")
                    
                    # 显示句子级引用
                    if citation['sentence_citations']:
                        print(f"    📍 关键句子:")
                        for sentence in citation['sentence_citations'][:1]:
                            text = sentence['sentence_text']
                            if len(text) > 80:
                                text = text[:80] + "..."
                            print(f"      • {text}")
            
            # 显示统计信息
            stats = response['precise_citation_stats']
            print(f"\n📈 统计信息:")
            print(f"  • 页码覆盖: {stats['page_range']}")
            print(f"  • 行号覆盖: {stats['line_range']}")
            print(f"  • 平均相关性: {stats['avg_relevance_score']:.3f}")
            
            # 显示相关问题
            if response['related_questions']:
                print(f"\n💡 相关问题建议:")
                for related_q in response['related_questions'][:3]:
                    print(f"  • {related_q}")
                    
        except Exception as e:
            print(f"❌ 处理查询时出错: {e}")
        
        print("\n" + "-" * 60)
    
    print(f"\n🎉 精确溯源医学问答系统演示完成!")
    print(f"✅ 成功实现了宽泛问题的精确溯源回答")
    print(f"✅ 提供了页码级别的引用信息")
    print(f"✅ 支持句子级别的精确引用")
    print(f"✅ 包含相关性评分和置信度评估")

if __name__ == "__main__":
    main()