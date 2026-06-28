#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医学教材问答系统完整演示
基于RAGFlow与DeepSeek的智能医学问答
"""

import os
import sys
import json
import logging
from datetime import datetime

# 添加scripts目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from vectorization_engine import MedicalVectorEngine
from deepseek_integration import MedicalQASystem

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalQADemo:
    """医学问答系统演示"""
    
    def __init__(self):
        """初始化演示系统"""
        self.vector_engine = MedicalVectorEngine()
        self.qa_system = MedicalQASystem()
        
        print("=== 医学教材问答系统演示 ===")
        print("基于RAGFlow与DeepSeek大模型")
        print("=" * 50)
    
    def demonstrate_knowledge_retrieval(self):
        """演示知识检索功能"""
        print("\n📚 知识检索演示")
        print("-" * 30)
        
        # 测试查询
        test_queries = [
            {
                "query": "心脏病的病因有哪些？",
                "description": "病因查询"
            },
            {
                "query": "冠心病的诊断标准是什么？",
                "description": "诊断查询"
            },
            {
                "query": "心力衰竭的治疗药物有哪些？",
                "description": "治疗查询"
            }
        ]
        
        for i, test_case in enumerate(test_queries, 1):
            query = test_case["query"]
            description = test_case["description"]
            
            print(f"\n{i}. {description}")
            print(f"   查询: {query}")
            print("   " + "-" * 40)
            
            # 执行检索
            results = self.vector_engine.search_similar_chunks(query, top_k=3)
            
            if results:
                print(f"   找到 {len(results)} 个相关知识块:")
                
                for j, result in enumerate(results, 1):
                    chunk = result["chunk"]
                    similarity = result["similarity"]
                    
                    print(f"   [{j}] 相似度: {similarity:.3f}")
                    print(f"       内容: {chunk['content'][:80]}...")
                    
                    # 显示元数据
                    metadata = chunk.get('metadata', {})
                    if metadata:
                        doc_title = metadata.get('document_title', 'N/A')
                        page_num = metadata.get('page_number', 'N/A')
                        print(f"       来源: 《{doc_title}》第{page_num}页")
                    
                    print()
            else:
                print("   未找到相关知识块")
    
    def demonstrate_qa_system(self):
        """演示问答系统"""
        print("\n🤖 智能问答演示")
        print("-" * 30)
        
        # 准备测试上下文（从检索结果中获取）
        sample_context = [
            {
                "content": "心脏疾病的病因主要包括先天性因素和后天性因素。先天性心脏病是由于胚胎期心脏发育异常所致，而后天性心脏病则多由感染、缺血、创伤等因素引起。",
                "metadata": {
                    "document_title": "心脏病学",
                    "page_number": "15"
                }
            },
            {
                "content": "冠心病的诊断需要结合临床表现、实验室检查和影像学检查。实验室检查包括心肌酶谱、血脂、血糖等指标。影像学检查主要有心电图、超声心动图、冠脉造影等。",
                "metadata": {
                    "document_title": "心脏病学",
                    "page_number": "28"
                }
            }
        ]
        
        # 测试问题
        test_questions = [
            "心脏病的病因主要有哪些？",
            "如何诊断冠心病？",
            "心力衰竭的治疗原则是什么？"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. 问题: {question}")
            print("   " + "-" * 40)
            
            # 使用QA系统回答（模拟，实际需要DeepSeek API）
            response = self.qa_system.create_fallback_response(question, sample_context)
            
            if response["success"]:
                print("   回答:")
                print(f"   {response['response']}")
                print(f"   置信度: {response.get('confidence_score', 'N/A')}")
                
                # 显示引用
                citations = response.get('citations', [])
                if citations:
                    print("   引用来源:")
                    for citation in citations:
                        print(f"   - 《{citation['document_title']}》第{citation['page_number']}页")
            else:
                print(f"   回答失败: {response.get('error', '未知错误')}")
    
    def demonstrate_system_capabilities(self):
        """演示系统能力"""
        print("\n⚡ 系统能力展示")
        print("-" * 30)
        
        capabilities = [
            {
                "feature": "医学知识分块",
                "description": "基于医学知识链的智能分块，保持医学知识完整性",
                "status": "✅ 已实现"
            },
            {
                "feature": "向量语义检索",
                "description": "基于BGE嵌入模型的语义检索，理解医学概念",
                "status": "✅ 已实现"
            },
            {
                "feature": "混合检索策略",
                "description": "向量检索+关键词检索的混合检索方式",
                "status": "🔧 配置完成"
            },
            {
                "feature": "DeepSeek大模型集成",
                "description": "集成DeepSeek API进行专业医学问答",
                "status": "✅ 代码完成"
            },
            {
                "feature": "答案溯源",
                "description": "提供精确的文献引用和页码信息",
                "status": "✅ 已实现"
            },
            {
                "feature": "医学术语处理",
                "description": "专业医学术语的识别和标准化处理",
                "status": "✅ 已实现"
            }
        ]
        
        for capability in capabilities:
            print(f"\n• {capability['feature']}")
            print(f"  描述: {capability['description']}")
            print(f"  状态: {capability['status']}")
    
    def show_system_statistics(self):
        """显示系统统计信息"""
        print("\n📊 系统统计信息")
        print("-" * 30)
        
        # 读取向量化统计
        stats_file = "data/vectorized/vectorization_stats.json"
        if os.path.exists(stats_file):
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            
            print(f"已处理文档数: {stats.get('total_documents', 0)}")
            print(f"已生成知识块数: {stats.get('total_chunks', 0)}")
            print(f"成功处理文档: {stats.get('successful_documents', 0)}")
            print(f"处理成功率: {stats.get('successful_documents', 0) / max(stats.get('total_documents', 1), 1) * 100:.1f}%")
        else:
            print("统计文件不存在")
        
        # 读取配置信息
        config_file = "configs/ragflow_config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            embedding_model = config.get('ragflow', {}).get('models', {}).get('embedding', {}).get('name', 'N/A')
            reranker_model = config.get('ragflow', {}).get('models', {}).get('reranker', {}).get('name', 'N/A')
            
            print(f"嵌入模型: {embedding_model}")
            print(f"重排序模型: {reranker_model}")
        
        print(f"演示时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def run_complete_demo(self):
        """运行完整演示"""
        print("🚀 启动医学问答系统演示...")
        
        # 1. 显示系统统计
        self.show_system_statistics()
        
        # 2. 演示知识检索
        self.demonstrate_knowledge_retrieval()
        
        # 3. 演示问答系统
        self.demonstrate_qa_system()
        
        # 4. 演示系统能力
        self.demonstrate_system_capabilities()
        
        print("\n" + "=" * 50)
        print("🎉 医学问答系统演示完成!")
        print("=" * 50)
        
        # 提供后续步骤建议
        print("\n📋 后续步骤:")
        print("1. 配置DeepSeek API密钥以启用完整问答功能")
        print("2. 导入更多医学PDF教材")
        print("3. 部署RAGFlow服务")
        print("4. 配置生产环境")
        print("5. 进行系统性能测试")

def main():
    """主函数"""
    try:
        demo = MedicalQADemo()
        demo.run_complete_demo()
        
    except Exception as e:
        logger.error(f"演示运行失败: {e}")
        print(f"❌ 演示运行失败: {e}")

if __name__ == "__main__":
    main()