#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地检索功能测试脚本
验证从本地加载的数据能否被正确检索
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# 添加scripts目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from local_data_loader import LocalDataLoader
from local_vector_database import LocalVectorDatabase

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocalRetrievalTester:
    """本地检索功能测试器"""
    
    def __init__(self, 
                 data_root: str = "D:\\Study\\py_program\\导出分块\\数据",
                 vector_db_dir: str = "./data/chroma_db"):
        """
        初始化测试器
        
        Args:
            data_root: 数据根目录
            vector_db_dir: 向量数据库目录
        """
        self.data_root = data_root
        self.vector_db_dir = vector_db_dir
        
        # 初始化组件
        self.data_loader = LocalDataLoader(data_root)
        self.vector_db = LocalVectorDatabase(vector_db_dir)
        
        logger.info("本地检索功能测试器初始化完成")
    
    def test_data_loading(self) -> Dict[str, Any]:
        """
        测试数据加载
        
        Returns:
            测试结果
        """
        print("\n" + "=" * 60)
        print("📂 测试 1/5: 数据加载")
        print("=" * 60)
        
        try:
            # 加载数据
            data = self.data_loader.load_all_data()
            
            print(f"\n✅ 数据加载成功")
            print(f"   📄 文档数量: {data['total_documents']}")
            print(f"   📝 Chunks数量: {data['total_chunks']}")
            print(f"   📊 有向量的chunks: {data['document_stats']['chunks_with_embedding']}")
            print(f"   📊 无向量的chunks: {data['document_stats']['chunks_without_embedding']}")
            
            # 显示文档列表
            print(f"\n📚 文档列表（前10个）:")
            for i, doc_name in enumerate(self.data_loader.get_document_list()[:10], 1):
                print(f"   {i}. {doc_name}")
            
            return {
                'success': True,
                'total_documents': data['total_documents'],
                'total_chunks': data['total_chunks'],
                'chunks_with_embedding': data['document_stats']['chunks_with_embedding'],
                'chunks_without_embedding': data['document_stats']['chunks_without_embedding']
            }
            
        except Exception as e:
            print(f"\n❌ 数据加载失败: {str(e)}")
            logger.error(f"数据加载失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_vector_database(self) -> Dict[str, Any]:
        """
        测试向量数据库
        
        Returns:
            测试结果
        """
        print("\n" + "=" * 60)
        print("💾 测试 2/5: 向量数据库")
        print("=" * 60)
        
        try:
            # 获取统计信息
            stats = self.vector_db.get_collection_stats()
            
            print(f"\n✅ 向量数据库状态正常")
            print(f"   📊 总chunks: {stats['total_chunks']}")
            print(f"   📚 集合名称: {stats['collection_name']}")
            print(f"   📁 持久化目录: {stats['persist_directory']}")
            print(f"   🔍 倒排索引大小: {stats['inverted_index_size']}")
            
            return {
                'success': True,
                'total_chunks': stats['total_chunks'],
                'collection_name': stats['collection_name'],
                'inverted_index_size': stats['inverted_index_size']
            }
            
        except Exception as e:
            print(f"\n❌ 向量数据库测试失败: {str(e)}")
            logger.error(f"向量数据库测试失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_vector_search(self) -> Dict[str, Any]:
        """
        测试向量检索
        
        Returns:
            测试结果
        """
        print("\n" + "=" * 60)
        print("🔍 测试 3/5: 向量检索")
        print("=" * 60)
        
        test_queries = [
            "高血压的诊断标准",
            "冠心病的治疗方法",
            "心力衰竭的症状"
        ]
        
        results = []
        
        for query in test_queries:
            print(f"\n🔎 查询: {query}")
            
            try:
                # 生成查询向量（使用hash向量）
                import hashlib
                query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
                query_embedding = []
                for i in range(0, len(query_hash), 8):
                    chunk = query_hash[i:i+8]
                    value = int(chunk, 16) / (16**8 - 1)
                    query_embedding.extend([value] * 256)
                query_embedding = query_embedding[:1024]
                
                # 向量检索
                search_results = self.vector_db.vector_search(query_embedding, top_k=3)
                
                print(f"   ✅ 找到 {len(search_results)} 个结果")
                
                for i, result in enumerate(search_results, 1):
                    metadata = result.get('metadata', {})
                    print(f"   [{i}] 文档: {metadata.get('_document_name', 'unknown')}")
                    print(f"       页码: {metadata.get('page', 'unknown')}")
                    print(f"       距离: {result.get('distance', 0):.4f}")
                
                results.append({
                    'query': query,
                    'success': True,
                    'result_count': len(search_results)
                })
                
            except Exception as e:
                print(f"   ❌ 检索失败: {str(e)}")
                results.append({
                    'query': query,
                    'success': False,
                    'error': str(e)
                })
        
        success_count = sum(1 for r in results if r['success'])
        print(f"\n📊 测试结果: {success_count}/{len(test_queries)} 成功")
        
        return {
            'success': success_count == len(test_queries),
            'total_queries': len(test_queries),
            'successful_queries': success_count,
            'results': results
        }
    
    def test_keyword_search(self) -> Dict[str, Any]:
        """
        测试关键词检索
        
        Returns:
            测试结果
        """
        print("\n" + "=" * 60)
        print("🔑 测试 4/5: 关键词检索")
        print("=" * 60)
        
        test_queries = [
            "高血压",
            "冠心病",
            "心力衰竭"
        ]
        
        results = []
        
        for query in test_queries:
            print(f"\n🔎 查询: {query}")
            
            try:
                # 关键词检索
                search_results = self.vector_db.keyword_search(query, top_k=3)
                
                print(f"   ✅ 找到 {len(search_results)} 个结果")
                
                for i, result in enumerate(search_results, 1):
                    metadata = result.get('metadata', {})
                    print(f"   [{i}] 文档: {metadata.get('_document_name', 'unknown')}")
                    print(f"       关键词分数: {result.get('keyword_score', 0)}")
                
                results.append({
                    'query': query,
                    'success': True,
                    'result_count': len(search_results)
                })
                
            except Exception as e:
                print(f"   ❌ 检索失败: {str(e)}")
                results.append({
                    'query': query,
                    'success': False,
                    'error': str(e)
                })
        
        success_count = sum(1 for r in results if r['success'])
        print(f"\n📊 测试结果: {success_count}/{len(test_queries)} 成功")
        
        return {
            'success': success_count == len(test_queries),
            'total_queries': len(test_queries),
            'successful_queries': success_count,
            'results': results
        }
    
    def test_hybrid_search(self) -> Dict[str, Any]:
        """
        测试混合检索
        
        Returns:
            测试结果
        """
        print("\n" + "=" * 60)
        print("🔀 测试 5/5: 混合检索")
        print("=" * 60)
        
        test_queries = [
            "高血压的诊断标准是什么？",
            "冠心病如何治疗？",
            "心力衰竭有哪些症状？"
        ]
        
        results = []
        
        for query in test_queries:
            print(f"\n🔎 查询: {query}")
            
            try:
                # 生成查询向量
                import hashlib
                query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
                query_embedding = []
                for i in range(0, len(query_hash), 8):
                    chunk = query_hash[i:i+8]
                    value = int(chunk, 16) / (16**8 - 1)
                    query_embedding.extend([value] * 256)
                query_embedding = query_embedding[:1024]
                
                # 混合检索
                search_results = self.vector_db.hybrid_search(
                    query_embedding=query_embedding,
                    query=query,
                    top_k=3,
                    vector_weight=0.7,
                    keyword_weight=0.3
                )
                
                print(f"   ✅ 找到 {len(search_results)} 个结果")
                
                for i, result in enumerate(search_results, 1):
                    metadata = result.get('metadata', {})
                    print(f"   [{i}] 文档: {metadata.get('_document_name', 'unknown')}")
                    print(f"       页码: {metadata.get('page', 'unknown')}")
                    print(f"       融合分数: {result.get('fused_score', 0):.4f}")
                    print(f"       向量分数: {result.get('vector_score', 0):.4f}")
                    print(f"       关键词分数: {result.get('keyword_score', 0):.4f}")
                
                results.append({
                    'query': query,
                    'success': True,
                    'result_count': len(search_results)
                })
                
            except Exception as e:
                print(f"   ❌ 检索失败: {str(e)}")
                results.append({
                    'query': query,
                    'success': False,
                    'error': str(e)
                })
        
        success_count = sum(1 for r in results if r['success'])
        print(f"\n📊 测试结果: {success_count}/{len(test_queries)} 成功")
        
        return {
            'success': success_count == len(test_queries),
            'total_queries': len(test_queries),
            'successful_queries': success_count,
            'results': results
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        运行所有测试
        
        Returns:
            所有测试结果
        """
        print("=" * 60)
        print("🧪 本地检索功能测试套件")
        print("=" * 60)
        
        test_results = {}
        
        # 测试1: 数据加载
        test_results['data_loading'] = self.test_data_loading()
        
        # 测试2: 向量数据库
        test_results['vector_database'] = self.test_vector_database()
        
        # 测试3: 向量检索
        test_results['vector_search'] = self.test_vector_search()
        
        # 测试4: 关键词检索
        test_results['keyword_search'] = self.test_keyword_search()
        
        # 测试5: 混合检索
        test_results['hybrid_search'] = self.test_hybrid_search()
        
        # 总结
        print("\n" + "=" * 60)
        print("📊 测试总结")
        print("=" * 60)
        
        total_tests = len(test_results)
        passed_tests = sum(1 for t in test_results.values() if t.get('success', False))
        
        print(f"\n总测试数: {total_tests}")
        print(f"通过测试: {passed_tests}")
        print(f"失败测试: {total_tests - passed_tests}")
        print(f"通过率: {passed_tests/total_tests*100:.1f}%")
        
        print(f"\n测试详情:")
        for test_name, result in test_results.items():
            status = "✅ 通过" if result.get('success', False) else "❌ 失败"
            print(f"   {test_name}: {status}")
            if not result.get('success', False):
                print(f"       错误: {result.get('error', 'Unknown error')}")
        
        # 保存测试报告
        self._save_test_report(test_results)
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'pass_rate': passed_tests/total_tests*100,
            'test_results': test_results
        }
    
    def _save_test_report(self, test_results: Dict[str, Any]):
        """
        保存测试报告
        
        Args:
            test_results: 测试结果
        """
        report_file = Path("data/test_report.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 测试报告已保存: {report_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='本地检索功能测试脚本')
    parser.add_argument('--data-root', 
                       default='D:\\Study\\py_program\\导出分块\\数据',
                       help='数据根目录')
    parser.add_argument('--vector-db-dir',
                       default='./data/chroma_db',
                       help='向量数据库目录')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = LocalRetrievalTester(
        data_root=args.data_root,
        vector_db_dir=args.vector_db_dir
    )
    
    # 运行所有测试
    results = tester.run_all_tests()
    
    # 退出码
    sys.exit(0 if results['pass_rate'] == 100 else 1)


if __name__ == "__main__":
    main()
