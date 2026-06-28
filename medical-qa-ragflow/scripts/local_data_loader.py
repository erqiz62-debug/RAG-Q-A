#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地数据加载器
从RagFlow导出的分块数据中加载所有chunk，构建本地知识库
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import hashlib

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocalDataLoader:
    """本地数据加载器"""
    
    def __init__(self, data_root: str = "D:\\Study\\py_program\\导出分块\\数据"):
        """
        初始化本地数据加载器
        
        Args:
            data_root: RagFlow导出数据的根目录
        """
        self.data_root = Path(data_root)
        self.all_chunks = []
        self.document_metadata = {}
        self.document_stats = {}
        
        logger.info(f"本地数据加载器初始化完成，数据根目录: {self.data_root}")
    
    def load_all_data(self) -> Dict[str, Any]:
        """
        加载所有数据
        
        Returns:
            包含所有chunks和元数据的字典
        """
        logger.info("开始加载所有数据...")
        
        # 遍历所有子文件夹
        subdirs = [d for d in self.data_root.iterdir() if d.is_dir()]
        logger.info(f"找到 {len(subdirs)} 个子文件夹")
        
        total_chunks = 0
        total_documents = 0
        
        for subdir in subdirs:
            chunks = self._load_document_chunks(subdir)
            if chunks:
                self.all_chunks.extend(chunks)
                total_chunks += len(chunks)
                total_documents += 1
        
        logger.info(f"数据加载完成: {total_documents} 个文档, {total_chunks} 个chunks")
        
        # 构建统计信息
        self._build_statistics()
        
        return {
            "chunks": self.all_chunks,
            "document_metadata": self.document_metadata,
            "document_stats": self.document_stats,
            "total_chunks": total_chunks,
            "total_documents": total_documents
        }
    
    def _load_document_chunks(self, doc_dir: Path) -> Optional[List[Dict[str, Any]]]:
        """
        加载单个文档的chunks
        
        Args:
            doc_dir: 文档文件夹路径
            
        Returns:
            chunks列表
        """
        chunks_file = doc_dir / "chunks.json"
        metadata_file = doc_dir / "metadata.json"
        
        if not chunks_file.exists():
            logger.warning(f"chunks.json不存在: {doc_dir}")
            return None
        
        try:
            # 加载chunks
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            # 去重：每个chunk_id只保留一个版本
            unique_chunks = {}
            for chunk in chunks:
                chunk_id = chunk.get('chunk_id', '')
                if chunk_id:
                    # 如果这个chunk_id已经存在，跳过
                    if chunk_id in unique_chunks:
                        continue
                    # 否则，保存这个chunk
                    unique_chunks[chunk_id] = chunk
            
            # 转换为列表
            chunks = list(unique_chunks.values())
            
            # 加载metadata
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 保存文档元数据
                doc_id = metadata.get('document_id', doc_dir.name)
                self.document_metadata[doc_id] = metadata
                
                # 为每个chunk添加文档信息
                for chunk in chunks:
                    chunk['_document_name'] = metadata.get('document_name', doc_dir.name)
                    chunk['_document_id'] = doc_id
                    chunk['_dataset_id'] = metadata.get('dataset_id', '')
                    
                    # 确保chunk_id是全局唯一的（添加文档前缀）
                    original_chunk_id = chunk.get('chunk_id', '')
                    if original_chunk_id:
                        chunk['chunk_id'] = f"{doc_id}_{original_chunk_id}"
                    else:
                        # 如果没有chunk_id，生成一个唯一的
                        chunk['chunk_id'] = f"{doc_id}_{chunk.get('chunk_index', 0)}"
                    
                    # 处理metadata字段
                    if 'metadata' not in chunk or not chunk['metadata']:
                        chunk['metadata'] = {}
                    
                    # 添加页码信息（如果存在）
                    if 'page' not in chunk['metadata']:
                        chunk['metadata']['page'] = chunk.get('chunk_index', 0)
                    
                    # 添加section信息
                    if 'section' not in chunk['metadata']:
                        chunk['metadata']['section'] = ''
                
                logger.info(f"加载文档: {metadata.get('document_name', doc_dir.name)}, {len(chunks)} 个chunks")
                return chunks
            
            else:
                logger.warning(f"metadata.json不存在: {doc_dir}")
                return None
                
        except Exception as e:
            logger.error(f"加载文档失败 {doc_dir}: {str(e)}")
            return None
    
    def _build_statistics(self):
        """构建统计信息"""
        logger.info("构建统计信息...")
        
        # 按文档统计
        doc_chunk_count = defaultdict(int)
        doc_word_count = defaultdict(int)
        
        for chunk in self.all_chunks:
            doc_name = chunk.get('_document_name', 'unknown')
            doc_chunk_count[doc_name] += 1
            
            # 统计字数
            content = chunk.get('content', '')
            doc_word_count[doc_name] += len(content)
        
        self.document_stats = {
            'chunk_count_by_doc': dict(doc_chunk_count),
            'word_count_by_doc': dict(doc_word_count),
            'total_chunks': len(self.all_chunks),
            'total_documents': len(self.document_metadata),
            'chunks_with_embedding': sum(1 for c in self.all_chunks if c.get('embedding') is not None),
            'chunks_without_embedding': sum(1 for c in self.all_chunks if c.get('embedding') is None)
        }
        
        logger.info(f"统计信息: {self.document_stats}")
    
    def get_chunks_by_document(self, document_name: str) -> List[Dict[str, Any]]:
        """
        获取指定文档的所有chunks
        
        Args:
            document_name: 文档名称
            
        Returns:
            chunks列表
        """
        return [c for c in self.all_chunks if c.get('_document_name') == document_name]
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        根据chunk_id获取chunk
        
        Args:
            chunk_id: chunk ID
            
        Returns:
            chunk字典或None
        """
        for chunk in self.all_chunks:
            if chunk.get('chunk_id') == chunk_id:
                return chunk
        return None
    
    def get_document_list(self) -> List[str]:
        """
        获取所有文档名称列表
        
        Returns:
            文档名称列表
        """
        return list(set(c.get('_document_name', '') for c in self.all_chunks))
    
    def export_to_json(self, output_path: str = "data/local_knowledge_base.json"):
        """
        导出为JSON文件
        
        Args:
            output_path: 输出文件路径
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            "metadata": {
                "created_at": "2026-02-15",
                "version": "2.0",
                "total_chunks": len(self.all_chunks),
                "total_documents": len(self.document_metadata),
                "source": "RagFlow导出数据",
                "data_root": str(self.data_root)
            },
            "document_metadata": self.document_metadata,
            "document_stats": self.document_stats,
            "chunks": self.all_chunks
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"知识库已导出到: {output_file}")
        return output_file


def main():
    """测试数据加载器"""
    print("=" * 60)
    print("本地数据加载器测试")
    print("=" * 60)
    
    # 创建数据加载器
    loader = LocalDataLoader()
    
    # 加载所有数据
    data = loader.load_all_data()
    
    print(f"\n📊 加载统计:")
    print(f"   文档数量: {data['total_documents']}")
    print(f"   Chunks数量: {data['total_chunks']}")
    print(f"   有向量的chunks: {data['document_stats']['chunks_with_embedding']}")
    print(f"   无向量的chunks: {data['document_stats']['chunks_without_embedding']}")
    
    print(f"\n📚 文档列表:")
    for i, doc_name in enumerate(loader.get_document_list()[:10], 1):
        print(f"   {i}. {doc_name}")
    
    print(f"\n💾 导出知识库...")
    loader.export_to_json()
    
    print("\n✅ 测试完成!")


if __name__ == "__main__":
    main()
