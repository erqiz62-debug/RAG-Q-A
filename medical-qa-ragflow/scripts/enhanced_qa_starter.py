#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版医学问答系统启动脚本
支持宽泛问题回答和精确溯源功能
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加脚本路径
scripts_dir = Path(__file__).parent
sys.path.append(str(scripts_dir))

from enhanced_document_parser import EnhancedMedicalDocumentParser
from enhanced_medical_qa_system import EnhancedMedicalQASystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_qa_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedMedicalQAStarter:
    def __init__(self):
        """初始化启动器"""
        self.project_root = Path("d:\\BS\\medical-qa-ragflow")
        self.pdf_directory = Path("d:\\BS\\教材")
        self.data_directory = self.project_root / "data"
        self.scripts_directory = self.project_root / "scripts"
        
        # 确保目录存在
        self.data_directory.mkdir(exist_ok=True)
        self.scripts_directory.mkdir(exist_ok=True)
        
        logger.info("增强版医学问答系统启动器初始化完成")
    
    def build_enhanced_knowledge_base(self):
        """构建增强版知识库"""
        logger.info("开始构建增强版知识库...")
        
        try:
            # 创建解析器
            parser = EnhancedMedicalDocumentParser(
                pdf_directory=str(self.pdf_directory),
                output_directory=str(self.data_directory)
            )
            
            # 构建知识库
            knowledge_base = parser.build_comprehensive_knowledge_base()
            
            logger.info("增强版知识库构建完成!")
            return knowledge_base
            
        except Exception as e:
            logger.error(f"构建知识库时出错: {e}")
            return None
    
    def test_broad_question_handling(self):
        """测试宽泛问题处理"""
        logger.info("测试宽泛问题处理...")
        
        try:
            # 加载知识库
            qa_system = EnhancedMedicalQASystem(
                knowledge_base_path=str(self.data_directory / "enhanced_medical_knowledge_base.json")
            )
            
            # 测试宽泛问题
            test_queries = [
                "什么是心脏病？",
                "高血压如何治疗？",
                "冠心病有哪些症状？",
                "为什么会有心律失常？",
                "心力衰竭的诊断方法",
                "心肌梗死的急救措施",
                "心绞痛的特征",
                "心脏病的预防措施"
            ]
            
            results = []
            
            for query in test_queries:
                logger.info(f"测试查询: {query}")
                
                response = qa_system.process_query(query)
                
                result = {
                    'query': query,
                    'answer': response['answer'],
                    'confidence': response['confidence'],
                    'citations_count': len(response['citations']),
                    'related_questions': response['related_questions']
                }
                
                results.append(result)
                
                # 打印结果
                print(f"\n{'='*60}")
                print(f"查询: {query}")
                print(f"置信度: {response['confidence']:.3f}")
                print(f"引用数量: {len(response['citations'])}")
                print(f"回答: {response['answer'][:200]}...")
                print(f"相关问题: {response['related_questions'][:3]}")
            
            # 保存测试结果
            test_results_path = self.data_directory / "broad_question_test_results.json"
            with open(test_results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"测试结果已保存到: {test_results_path}")
            return results
            
        except Exception as e:
            logger.error(f"测试宽泛问题时出错: {e}")
            return None
    
    def demonstrate_precise_citations(self):
        """演示精确溯源功能"""
        logger.info("演示精确溯源功能...")
        
        try:
            qa_system = EnhancedMedicalQASystem(
                knowledge_base_path=str(self.data_directory / "enhanced_medical_knowledge_base.json")
            )
            
            # 选择一个具体查询来演示
            demo_query = "什么是冠心病？"
            logger.info(f"演示查询: {demo_query}")
            
            response = qa_system.process_query(demo_query)
            
            print(f"\n{'='*60}")
            print(f"精确溯源演示")
            print(f"{'='*60}")
            print(f"查询: {demo_query}")
            print(f"回答: {response['answer']}")
            print(f"置信度: {response['confidence']:.3f}")
            print(f"引用信息:")
            
            for i, citation in enumerate(response['citations'], 1):
                print(f"\n引用 {i}:")
                print(f"  源文件: {citation['source_file']}")
                print(f"  页码: 第 {citation['page_number']} 页")
                print(f"  知识块: {citation['block_index']}")
                print(f"  知识类型: {citation['knowledge_types']}")
                print(f"  医学术语: {citation['medical_terms']}")
                if 'key_sentence' in citation:
                    print(f"  关键句子: {citation['key_sentence']}")
                print(f"  相关性分数: {citation['relevance_score']:.3f}")
            
            return response
            
        except Exception as e:
            logger.error(f"演示精确溯源时出错: {e}")
            return None
    
    def generate_system_report(self):
        """生成系统报告"""
        logger.info("生成系统报告...")
        
        try:
            # 加载知识库统计信息
            knowledge_base_path = self.data_directory / "enhanced_medical_knowledge_base.json"
            if knowledge_base_path.exists():
                with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                    knowledge_base = json.load(f)
                
                metadata = knowledge_base.get('metadata', {})
                statistics = knowledge_base.get('statistics', {})
                
                report = {
                    'system_info': {
                        'name': '增强版医学问答系统',
                        'version': '2.0_enhanced',
                        'created_at': '2025-12-26',
                        'capabilities': [
                            '宽泛问题智能理解',
                            '精确溯源功能',
                            '医学术语识别',
                            '多知识类型支持',
                            '同义词扩展',
                            '置信度评估'
                        ]
                    },
                    'knowledge_base_stats': {
                        'total_chunks': metadata.get('total_chunks', 0),
                        'processed_files': len(metadata.get('processed_files', [])),
                        'medical_terms_count': statistics.get('medical_terms_count', 0),
                        'knowledge_types': metadata.get('knowledge_types', []),
                        'medical_categories': len(metadata.get('medical_categories', []))
                    },
                    'features': {
                        'broad_question_handling': '支持"什么是"、"如何治疗"、"为什么"等宽泛问题',
                        'precise_citation': '提供精确的页码、句子级溯源信息',
                        'query_expansion': '智能扩展查询以提高检索覆盖率',
                        'confidence_scoring': '多维度置信度评估',
                        'related_questions': '基于回答自动生成相关问题建议'
                    },
                    'usage_guide': {
                        'supported_query_types': [
                            '什么是XXX？',
                            '如何治疗XXX？',
                            'XXX的症状有哪些？',
                            '为什么会出现XXX？',
                            'XXX的诊断方法',
                            'XXX的预防措施'
                        ],
                        'citation_format': '源文件 + 页码 + 知识块 + 关键句子',
                        'confidence_levels': {
                            'high': '0.7-1.0 (高度相关)',
                            'medium': '0.4-0.7 (中度相关)',
                            'low': '0.1-0.4 (低度相关)'
                        }
                    }
                }
                
                # 保存报告
                report_path = self.data_directory / "enhanced_system_report.json"
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                
                logger.info(f"系统报告已保存到: {report_path}")
                return report
            
        except Exception as e:
            logger.error(f"生成系统报告时出错: {e}")
            return None
    
    def run_full_demo(self):
        """运行完整演示"""
        logger.info("开始完整演示...")
        
        print(f"{'='*80}")
        print(f"增强版医学问答系统演示")
        print(f"支持宽泛问题回答和精确溯源功能")
        print(f"{'='*80}")
        
        # 1. 构建知识库
        logger.info("步骤 1: 构建增强版知识库")
        knowledge_base = self.build_enhanced_knowledge_base()
        if not knowledge_base:
            logger.error("知识库构建失败")
            return False
        
        print(f"✓ 知识库构建完成，包含 {knowledge_base['metadata']['total_chunks']} 个知识块")
        
        # 2. 测试宽泛问题处理
        logger.info("步骤 2: 测试宽泛问题处理")
        test_results = self.test_broad_question_handling()
        if test_results:
            print(f"✓ 宽泛问题测试完成，测试了 {len(test_results)} 个查询")
            
            # 显示测试统计
            avg_confidence = sum(r['confidence'] for r in test_results) / len(test_results)
            total_citations = sum(r['citations_count'] for r in test_results)
            print(f"  平均置信度: {avg_confidence:.3f}")
            print(f"  总引用数量: {total_citations}")
        
        # 3. 演示精确溯源
        logger.info("步骤 3: 演示精确溯源功能")
        citation_demo = self.demonstrate_precise_citations()
        if citation_demo:
            print("✓ 精确溯源功能演示完成")
            print(f"  引用信息包含: 页码、句子、知识类型")
        
        # 4. 生成系统报告
        logger.info("步骤 4: 生成系统报告")
        report = self.generate_system_report()
        if report:
            print("✓ 系统报告生成完成")
            print(f"  支持的查询类型: {len(report['usage_guide']['supported_query_types'])}")
        
        print(f"\n{'='*80}")
        print(f"演示完成！系统现在支持：")
        print(f"1. 宽泛医学问题的智能理解和回答")
        print(f"2. 精确到页码和句子的溯源信息")
        print(f"3. 多维度置信度评估")
        print(f"4. 智能相关问题推荐")
        print(f"{'='*80}")
        
        return True

def main():
    """主函数"""
    print("启动增强版医学问答系统...")
    
    starter = EnhancedMedicalQAStarter()
    
    # 运行完整演示
    success = starter.run_full_demo()
    
    if success:
        print("\n系统已准备就绪！您现在可以：")
        print("1. 使用增强版问答系统回答更宽泛的医学问题")
        print("2. 获得精确到页码和句子的溯源信息")
        print("3. 查看相关问题推荐")
        print("4. 参考置信度评分评估答案可靠性")
    else:
        print("\n系统启动过程中遇到问题，请检查日志文件。")

if __name__ == "__main__":
    main()