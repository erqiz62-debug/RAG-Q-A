#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地知识库构建器
从RagFlow导出的数据构建本地知识库，完全脱离RagFlow依赖
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加scripts目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from local_data_loader import LocalDataLoader
from local_vector_database import LocalVectorDatabase

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocalKnowledgeBaseBuilder:
    """本地知识库构建器"""
    
    def __init__(self, 
                 data_root: str = "D:\\Study\\py_program\\导出分块\\数据",
                 vector_db_dir: str = "./data/chroma_db",
                 collection_name: str = "medical_knowledge"):
        """
        初始化知识库构建器
        
        Args:
            data_root: RagFlow导出数据根目录
            vector_db_dir: 向量数据库持久化目录
            collection_name: 集合名称
        """
        self.data_root = data_root
        self.vector_db_dir = vector_db_dir
        self.collection_name = collection_name
        
        # 初始化组件
        self.data_loader = LocalDataLoader(data_root)
        self.vector_db = LocalVectorDatabase(vector_db_dir, collection_name)
        
        # 嵌入模型（如果需要重新生成向量）
        self.embedding_model = None
        
        logger.info("本地知识库构建器初始化完成")
    
    def build_knowledge_base(self, 
                          regenerate_embeddings: bool = False,
                          embedding_model_name: str = "BAAI/bge-large-zh"):
        """
        构建知识库
        
        Args:
            regenerate_embeddings: 是否重新生成embedding
            embedding_model_name: 嵌入模型名称
        """
        logger.info("开始构建本地知识库...")
        print("=" * 60)
        print("🏥 本地医学知识库构建器")
        print("=" * 60)
        
        # 步骤1: 加载所有数据
        print(f"\n📂 步骤 1/4: 加载数据")
        print(f"   数据根目录: {self.data_root}")
        
        data = self.data_loader.load_all_data()
        
        print(f"   ✅ 加载完成")
        print(f"   📄 文档数量: {data['total_documents']}")
        print(f"   📝 Chunks数量: {data['total_chunks']}")
        print(f"   📊 有向量的chunks: {data['document_stats']['chunks_with_embedding']}")
        print(f"   📊 无向量的chunks: {data['document_stats']['chunks_without_embedding']}")
        
        # 步骤2: 处理embedding
        print(f"\n🔢 步骤 2/4: 处理向量")
        
        if regenerate_embeddings:
            print(f"   ⚠️  需要重新生成所有embeddings")
            print(f"   📦 模型: {embedding_model_name}")
            
            # 加载嵌入模型
            self.embedding_model = self._load_embedding_model(embedding_model_name)
            
            if self.embedding_model is None:
                print(f"   ❌ 无法加载嵌入模型，使用现有embeddings")
                regenerate_embeddings = False
            else:
                # 为没有embedding的chunks生成向量
                chunks_without_embedding = [
                    c for c in data['chunks'] if c.get('embedding') is None
                ]
                
                if chunks_without_embedding:
                    print(f"   🔄 为 {len(chunks_without_embedding)} 个chunks生成embeddings...")
                    self._generate_embeddings_for_chunks(chunks_without_embedding)
                else:
                    print(f"   ✅ 所有chunks都有embeddings")
        else:
            print(f"   ✅ 使用现有embeddings")
        
        # 步骤3: 添加到向量数据库
        print(f"\n💾 步骤 3/4: 构建向量数据库")
        print(f"   📁 向量数据库目录: {self.vector_db_dir}")
        print(f"   📚 集合名称: {self.collection_name}")
        
        # 重置向量数据库
        self.vector_db.reset()
        
        # 添加chunks
        self.vector_db.add_chunks(
            data['chunks'],
            embedding_model=self.embedding_model if regenerate_embeddings else None
        )
        
        # 获取统计信息
        stats = self.vector_db.get_collection_stats()
        print(f"   ✅ 向量数据库构建完成")
        print(f"   📊 总chunks: {stats['total_chunks']}")
        print(f"   🔍 倒排索引大小: {stats['inverted_index_size']}")
        
        # 步骤4: 导出知识库
        print(f"\n💾 步骤 4/4: 导出知识库")
        
        output_file = self.data_loader.export_to_json(
            "data/local_knowledge_base.json"
        )
        
        print(f"   ✅ 知识库已导出: {output_file}")
        
        # 完成总结
        print(f"\n" + "=" * 60)
        print("✅ 知识库构建完成!")
        print("=" * 60)
        print(f"📊 构建统计:")
        print(f"   文档数量: {data['total_documents']}")
        print(f"   Chunks数量: {data['total_chunks']}")
        print(f"   向量数据库: {stats['total_chunks']} 个向量")
        print(f"   倒排索引: {stats['inverted_index_size']} 个关键词")
        print(f"   知识库文件: {output_file}")
        print(f"\n💡 下一步:")
        print(f"   1. 运行测试脚本验证检索功能")
        print(f"   2. 启动Flask应用: python app.py")
        print(f"   3. 访问Web界面: http://localhost:5001")
        print("=" * 60)
        
        return {
            'success': True,
            'total_documents': data['total_documents'],
            'total_chunks': data['total_chunks'],
            'vector_db_stats': stats,
            'knowledge_base_file': str(output_file)
        }
    
    def _load_embedding_model(self, model_name: str):
        """
        加载嵌入模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            嵌入模型对象或None
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            print(f"   📦 加载嵌入模型: {model_name}")
            model = SentenceTransformer(model_name)
            print(f"   ✅ 模型加载成功")
            return model
            
        except ImportError:
            print(f"   ❌ sentence-transformers未安装")
            print(f"   💡 安装命令: pip install sentence-transformers")
            return None
        except Exception as e:
            print(f"   ❌ 加载模型失败: {e}")
            return None
    
    def _generate_embeddings_for_chunks(self, chunks: List[Dict[str, Any]]):
        """
        为chunks生成embeddings
        
        Args:
            chunks: chunks列表
        """
        import time
        
        batch_size = 32
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            contents = [c.get('content', '') for c in batch]
            
            try:
                # 批量生成embeddings
                embeddings = self.embedding_model.encode(
                    contents,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                
                # 更新chunks
                for j, chunk in enumerate(batch):
                    chunk['embedding'] = embeddings[j].tolist()
                
                print(f"   🔄 处理进度: {min(i + batch_size, total_chunks)}/{total_chunks}")
                
            except Exception as e:
                logger.error(f"生成embeddings失败: {e}")
                continue
    
    def update_knowledge_base(self, new_data_root: Optional[str] = None):
        """
        更新知识库（增量更新）
        
        Args:
            new_data_root: 新数据根目录（如果None则使用现有目录）
        """
        logger.info("开始更新知识库...")
        
        if new_data_root:
            self.data_root = new_data_root
            self.data_loader = LocalDataLoader(new_data_root)
        
        # 加载新数据
        data = self.data_loader.load_all_data()
        
        # 添加到向量数据库（不重置）
        self.vector_db.add_chunks(data['chunks'])
        
        # 导出更新后的知识库
        output_file = self.data_loader.export_to_json(
            "data/local_knowledge_base.json"
        )
        
        logger.info(f"知识库更新完成: {output_file}")
        
        return {
            'success': True,
            'new_chunks': len(data['chunks']),
            'knowledge_base_file': str(output_file)
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='本地医学知识库构建器')
    parser.add_argument('--data-root', 
                       default='D:\\Study\\py_program\\导出分块\\数据',
                       help='RagFlow导出数据根目录')
    parser.add_argument('--vector-db-dir',
                       default='./data/chroma_db',
                       help='向量数据库持久化目录')
    parser.add_argument('--collection-name',
                       default='medical_knowledge',
                       help='ChromaDB集合名称')
    parser.add_argument('--regenerate-embeddings',
                       action='store_true',
                       help='重新生成所有embeddings')
    parser.add_argument('--embedding-model',
                       default='BAAI/bge-large-zh',
                       help='嵌入模型名称')
    parser.add_argument('--update',
                       action='store_true',
                       help='增量更新知识库')
    
    args = parser.parse_args()
    
    # 创建构建器
    builder = LocalKnowledgeBaseBuilder(
        data_root=args.data_root,
        vector_db_dir=args.vector_db_dir,
        collection_name=args.collection_name
    )
    
    # 构建或更新知识库
    if args.update:
        result = builder.update_knowledge_base()
    else:
        result = builder.build_knowledge_base(
            regenerate_embeddings=args.regenerate_embeddings,
            embedding_model_name=args.embedding_model
        )
    
    # 保存构建报告
    report_file = Path("data/build_report.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 构建报告已保存: {report_file}")


if __name__ == "__main__":
    main()
