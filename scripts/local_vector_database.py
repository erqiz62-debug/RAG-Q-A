#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地向量数据库引擎
使用ChromaDB存储和检索向量，支持混合检索（向量+关键词）
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import hashlib

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB未安装，将使用内存向量存储")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocalVectorDatabase:
    """本地向量数据库"""
    
    def __init__(self, persist_directory: str = "./data/chroma_db", 
                 collection_name: str = "medical_knowledge"):
        """
        初始化本地向量数据库
        
        Args:
            persist_directory: ChromaDB持久化目录
            collection_name: 集合名称
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chroma_client = None
        self.collection = None
        
        # 内存向量存储（备用方案）
        self.memory_vectors = {}
        self.memory_metadata = {}
        
        # 倒排索引（用于关键词检索）
        self.inverted_index = defaultdict(list)
        
        # 初始化向量数据库
        self._initialize_vector_db()
        
        logger.info(f"本地向量数据库初始化完成，集合: {collection_name}")
    
    def _initialize_vector_db(self):
        """初始化向量数据库"""
        if CHROMADB_AVAILABLE:
            try:
                # 创建ChromaDB客户端
                os.makedirs(self.persist_directory, exist_ok=True)
                self.chroma_client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                
                # 获取或创建集合
                try:
                    self.collection = self.chroma_client.get_collection(name=self.collection_name)
                    logger.info(f"加载现有集合: {self.collection_name}")
                except:
                    self.collection = self.chroma_client.create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info(f"创建新集合: {self.collection_name}")
                
                logger.info("ChromaDB初始化成功")
                
            except Exception as e:
                logger.error(f"ChromaDB初始化失败: {e}，使用内存存储")
                self._initialize_memory_storage()
        else:
            logger.warning("ChromaDB未安装，使用内存向量存储")
            self._initialize_memory_storage()
    
    def _initialize_memory_storage(self):
        """初始化内存向量存储"""
        self.memory_vectors = {}
        self.memory_metadata = {}
        logger.info("内存向量存储初始化完成")
    
    def add_chunks(self, chunks: List[Dict[str, Any]], 
                  embedding_model: Optional[Any] = None):
        """
        添加chunks到向量数据库
        
        Args:
            chunks: chunks列表
            embedding_model: 嵌入模型（如果chunks没有embedding字段）
        """
        logger.info(f"开始添加 {len(chunks)} 个chunks到向量数据库...")
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        skipped_count = 0
        added_count = 0
        
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id', '')
            content = chunk.get('content', '')
            metadata = chunk.get('metadata', {})
            
            # 检查chunk_id是否有效
            if not chunk_id:
                skipped_count += 1
                continue
            
            # 添加文档信息到metadata
            metadata['_document_name'] = chunk.get('_document_name', '')
            metadata['_document_id'] = chunk.get('_document_id', '')
            metadata['_chunk_id'] = chunk_id
            
            # 获取或生成embedding
            embedding = chunk.get('embedding')
            if embedding is None and embedding_model is not None:
                # 需要生成embedding
                embedding = self._generate_embedding(content, embedding_model)
            elif embedding is None:
                # 使用hash向量作为备用方案
                embedding = self._generate_hash_embedding(content)
            
            # 总是添加到向量数据库（即使embedding是hash生成的）
            if embedding is not None:
                ids.append(chunk_id)
                embeddings.append(embedding)
                documents.append(content)
                metadatas.append(metadata)
                added_count += 1
                
                # 构建倒排索引
                self._build_inverted_index(chunk_id, content, metadata)
        
        logger.info(f"处理完成: 添加 {added_count} 个chunks, 跳过 {skipped_count} 个chunks")
        
        # 添加到向量数据库
        if CHROMADB_AVAILABLE and self.collection:
            if embeddings:
                try:
                    logger.info(f"准备添加 {len(ids)} 个chunks到ChromaDB")
                    logger.info(f"IDs数量: {len(ids)}, Embeddings数量: {len(embeddings)}, Documents数量: {len(documents)}, Metadatas数量: {len(metadatas)}")
                    
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas
                    )
                    logger.info(f"成功添加 {len(ids)} 个chunks到ChromaDB")
                    
                    # 验证添加是否成功
                    count = self.collection.count()
                    logger.info(f"ChromaDB集合中的文档总数: {count}")
                    
                except Exception as e:
                    logger.error(f"添加到ChromaDB失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.warning("没有embeddings可添加到ChromaDB")
        else:
            # 使用内存存储
            for i, chunk_id in enumerate(ids):
                self.memory_vectors[chunk_id] = embeddings[i]
                self.memory_metadata[chunk_id] = metadatas[i]
            logger.info(f"成功添加 {len(ids)} 个chunks到内存存储")
        
        logger.info(f"倒排索引构建完成，包含 {len(self.inverted_index)} 个关键词")
    
    def _build_inverted_index(self, chunk_id: str, content: str, metadata: Dict):
        """
        构建倒排索引
        
        Args:
            chunk_id: chunk ID
            content: 文本内容
            metadata: 元数据
        """
        import jieba
        
        # 分词
        words = jieba.cut(content)
        
        # 过滤停用词和短词
        stop_words = {'的', '是', '在', '和', '或', '及', '与', '等', '包括', 
                    '主要', '可以', '应该', '需要', '一个', '这个', '那个', '有'}
        
        for word in words:
            word = word.strip()
            if len(word) >= 2 and word not in stop_words:
                self.inverted_index[word].append(chunk_id)
    
    def _generate_embedding(self, text: str, embedding_model: Any) -> List[float]:
        """
        生成文本嵌入向量
        
        Args:
            text: 文本内容
            embedding_model: 嵌入模型
            
        Returns:
            向量列表
        """
        try:
            # 使用嵌入模型生成向量
            if hasattr(embedding_model, 'encode'):
                embedding = embedding_model.encode(text)
                return embedding.tolist()
            else:
                logger.error("嵌入模型没有encode方法")
                return None
        except Exception as e:
            logger.error(f"生成embedding失败: {e}")
            return None
    
    def _generate_hash_embedding(self, text: str) -> List[float]:
        """
        使用hash生成向量（备用方案）
        
        Args:
            text: 文本内容
            
        Returns:
            向量列表
        """
        import hashlib
        
        # 基于文本内容生成稳定的hash向量
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # 转换为1024维向量
        vector = []
        for i in range(0, len(text_hash), 8):
            chunk = text_hash[i:i+8]
            value = int(chunk, 16) / (16**8 - 1)
            vector.extend([value] * 256)  # 扩展到1024维
        
        return vector[:1024]  # 确保是1024维
    
    def vector_search(self, query_embedding: List[float], 
                   top_k: int = 10) -> List[Dict[str, Any]]:
        """
        向量检索
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        if CHROMADB_AVAILABLE and self.collection:
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
                
                # 格式化结果
                formatted_results = []
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'chunk_id': results['ids'][0][i],
                        'distance': results['distances'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'document': results['documents'][0][i]
                    })
                
                return formatted_results
                
            except Exception as e:
                logger.error(f"ChromaDB检索失败: {e}")
                return self._memory_vector_search(query_embedding, top_k)
        else:
            return self._memory_vector_search(query_embedding, top_k)
    
    def _memory_vector_search(self, query_embedding: List[float], 
                           top_k: int = 10) -> List[Dict[str, Any]]:
        """
        内存向量检索（备用方案）
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        results = []
        
        for chunk_id, embedding in self.memory_vectors.items():
            # 计算余弦相似度
            similarity = self._cosine_similarity(query_embedding, embedding)
            results.append({
                'chunk_id': chunk_id,
                'distance': 1 - similarity,  # 转换为距离
                'metadata': self.memory_metadata.get(chunk_id, {}),
                'similarity': similarity
            })
        
        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            相似度值
        """
        vec1_array = np.array(vec1)
        vec2_array = np.array(vec2)
        
        dot_product = np.dot(vec1_array, vec2_array)
        norm1 = np.linalg.norm(vec1_array)
        norm2 = np.linalg.norm(vec2_array)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        关键词检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        import jieba
        
        # 分词
        query_words = list(jieba.cut(query))
        
        # 统计每个chunk的匹配分数
        chunk_scores = defaultdict(int)
        
        for word in query_words:
            word = word.strip()
            if word in self.inverted_index:
                for chunk_id in self.inverted_index[word]:
                    chunk_scores[chunk_id] += 1
        
        # 获取top-k结果
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk_id, score in sorted_chunks[:top_k]:
            metadata = {}
            if CHROMADB_AVAILABLE and self.collection:
                try:
                    chunk_data = self.collection.get(ids=[chunk_id])
                    if chunk_data['metadatas']:
                        metadata = chunk_data['metadatas'][0]
                except:
                    pass
            else:
                metadata = self.memory_metadata.get(chunk_id, {})
            
            results.append({
                'chunk_id': chunk_id,
                'keyword_score': score,
                'metadata': metadata
            })
        
        return results
    
    def hybrid_search(self, query_embedding: List[float], query: str,
                   top_k: int = 10, vector_weight: float = 0.7,
                   keyword_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        混合检索（向量+关键词）
        
        Args:
            query_embedding: 查询向量
            query: 查询文本
            top_k: 返回结果数量
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
            
        Returns:
            检索结果列表
        """
        # 向量检索
        vector_results = self.vector_search(query_embedding, top_k=top_k * 2)
        
        # 关键词检索
        keyword_results = self.keyword_search(query, top_k=top_k * 2)
        
        # 融合结果
        fused_results = self._fuse_results(
            vector_results, keyword_results, 
            vector_weight, keyword_weight
        )
        
        # 返回top-k
        return fused_results[:top_k]
    
    def _fuse_results(self, vector_results: List[Dict], 
                    keyword_results: List[Dict],
                    vector_weight: float, keyword_weight: float) -> List[Dict]:
        """
        融合向量检索和关键词检索结果
        
        Args:
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
            vector_weight: 向量权重
            keyword_weight: 关键词权重
            
        Returns:
            融合后的结果列表
        """
        # 创建chunk_id到结果的映射
        vector_map = {r['chunk_id']: r for r in vector_results}
        keyword_map = {r['chunk_id']: r for r in keyword_results}
        
        # 获取所有唯一的chunk_id
        all_chunk_ids = set(vector_map.keys()) | set(keyword_map.keys())
        
        fused_results = []
        for chunk_id in all_chunk_ids:
            vector_score = 0.0
            keyword_score = 0.0
            
            if chunk_id in vector_map:
                # 将距离转换为相似度
                distance = vector_map[chunk_id].get('distance', 1.0)
                vector_score = 1.0 - distance
            
            if chunk_id in keyword_map:
                keyword_score = keyword_map[chunk_id].get('keyword_score', 0)
            
            # 归一化分数
            max_vector_score = max([1.0 - r.get('distance', 1.0) for r in vector_results], default=1.0)
            max_keyword_score = max([r.get('keyword_score', 0) for r in keyword_results], default=1.0)
            
            normalized_vector = vector_score / max_vector_score if max_vector_score > 0 else 0
            normalized_keyword = keyword_score / max_keyword_score if max_keyword_score > 0 else 0
            
            # 计算融合分数
            fused_score = (normalized_vector * vector_weight + 
                         normalized_keyword * keyword_weight)
            
            # 获取metadata
            metadata = {}
            if chunk_id in vector_map:
                metadata = vector_map[chunk_id].get('metadata', {})
            elif chunk_id in keyword_map:
                metadata = keyword_map[chunk_id].get('metadata', {})
            
            fused_results.append({
                'chunk_id': chunk_id,
                'fused_score': fused_score,
                'vector_score': normalized_vector,
                'keyword_score': normalized_keyword,
                'metadata': metadata
            })
        
        # 按融合分数排序
        fused_results.sort(key=lambda x: x['fused_score'], reverse=True)
        
        return fused_results
    
    def get_chunk_content(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        获取chunk的完整内容
        
        Args:
            chunk_id: chunk ID
            
        Returns:
            chunk字典或None
        """
        if CHROMADB_AVAILABLE and self.collection:
            try:
                chunk_data = self.collection.get(ids=[chunk_id])
                if chunk_data['documents']:
                    return {
                        'chunk_id': chunk_id,
                        'content': chunk_data['documents'][0],
                        'metadata': chunk_data['metadatas'][0]
                    }
            except Exception as e:
                logger.error(f"获取chunk内容失败: {e}")
                return None
        else:
            # 从内存存储获取
            metadata = self.memory_metadata.get(chunk_id)
            if metadata:
                return {
                    'chunk_id': chunk_id,
                    'content': '',  # 内存存储没有保存content
                    'metadata': metadata
                }
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            统计信息字典
        """
        if CHROMADB_AVAILABLE and self.collection:
            try:
                count = self.collection.count()
                return {
                    'total_chunks': count,
                    'collection_name': self.collection_name,
                    'persist_directory': self.persist_directory,
                    'inverted_index_size': len(self.inverted_index)
                }
            except Exception as e:
                logger.error(f"获取统计信息失败: {e}")
                return {
                    'total_chunks': len(self.memory_vectors),
                    'collection_name': self.collection_name,
                    'persist_directory': self.persist_directory,
                    'inverted_index_size': len(self.inverted_index)
                }
        else:
            return {
                'total_chunks': len(self.memory_vectors),
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory,
                'inverted_index_size': len(self.inverted_index)
            }
    
    def reset(self):
        """重置向量数据库"""
        if CHROMADB_AVAILABLE and self.collection:
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("向量数据库已重置")
            except Exception as e:
                logger.error(f"重置向量数据库失败: {e}")
        
        # 重置内存存储
        self.memory_vectors = {}
        self.memory_metadata = {}
        self.inverted_index = defaultdict(list)
        logger.info("内存存储已重置")


def main():
    """测试向量数据库"""
    print("=" * 60)
    print("本地向量数据库测试")
    print("=" * 60)
    
    # 创建向量数据库
    vector_db = LocalVectorDatabase()
    
    # 添加测试数据
    test_chunks = [
        {
            'chunk_id': 'test1',
            'content': '心脏病是威胁人类健康的主要疾病之一',
            'metadata': {'page': 1, 'section': '第一章'},
            '_document_name': 'test.pdf',
            '_document_id': 'doc1',
            'embedding': [0.1, 0.2, 0.3, 0.4]
        },
        {
            'chunk_id': 'test2',
            'content': '高血压的诊断标准是收缩压≥140mmHg',
            'metadata': {'page': 2, 'section': '第二章'},
            '_document_name': 'test.pdf',
            '_document_id': 'doc1',
            'embedding': [0.2, 0.3, 0.4, 0.5]
        }
    ]
    
    vector_db.add_chunks(test_chunks)
    
    # 获取统计信息
    stats = vector_db.get_collection_stats()
    print(f"\n📊 统计信息:")
    print(f"   总chunks: {stats['total_chunks']}")
    print(f"   倒排索引大小: {stats['inverted_index_size']}")
    
    # 测试检索
    print(f"\n🔍 测试检索:")
    query_embedding = [0.15, 0.25, 0.35, 0.45]
    results = vector_db.vector_search(query_embedding, top_k=2)
    
    for i, result in enumerate(results, 1):
        print(f"   [{i}] Chunk ID: {result['chunk_id']}")
        print(f"       距离: {result['distance']:.4f}")
        print(f"       文档: {result['metadata'].get('_document_name', 'unknown')}")
    
    print("\n✅ 测试完成!")


if __name__ == "__main__":
    main()
