#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医学教材问答系统 - 向量化引擎
实现医学知识块的向量化存储和检索
"""

import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import hashlib

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalVectorEngine:
    """医学向量数据库引擎"""
    
    def __init__(self, config_path: str = "configs/ragflow_config.json"):
        """初始化向量引擎"""
        self.config = self._load_config(config_path)
        self.embedding_model = None
        self.vector_db = None
        self.chroma_client = None
        
        # 医学特定配置
        self.medical_config = self._load_medical_config()
        
        logger.info("医学向量引擎初始化完成")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载RAGFlow配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_path} 未找到，使用默认配置")
            return self._get_default_config()
    
    def _load_medical_config(self) -> Dict:
        """加载医学特定配置"""
        try:
            with open("configs/medical_chunking_config.json", 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("医学分块配置未找到，使用默认配置")
            return self._get_default_medical_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "embedding_model": {
                "name": "bge-large-zh",
                "dimension": 1024,
                "max_seq_length": 512
            },
            "vector_db": {
                "type": "chroma",
                "collection_name": "medical_knowledge",
                "distance_metric": "cosine"
            },
            "retrieval": {
                "top_k": 5,
                "similarity_threshold": 0.7
            }
        }
    
    def _get_default_medical_config(self) -> Dict:
        """获取默认医学配置"""
        return {
            "chunking_strategy": "medical_knowledge_preservation",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "medical_patterns": ["病因", "病理", "临床表现", "诊断", "治疗", "预后"],
            "preserve_knowledge_chains": True
        }
    
    def initialize_embedding_model(self):
        """初始化嵌入模型"""
        try:
            # 模拟嵌入模型初始化
            # 实际使用中需要安装: pip install sentence-transformers
            logger.info("正在初始化嵌入模型...")
            
            # 这里使用模拟的嵌入向量生成
            self.embedding_model = "bge-large-zh-mock"
            
            logger.info(f"嵌入模型 {self.embedding_model} 初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            return False
    
    def initialize_vector_database(self):
        """初始化向量数据库"""
        try:
            # 模拟向量数据库初始化
            # 实际使用中需要安装: pip install chromadb
            db_type = self.config.get("vector_db", {}).get("type", "chroma")
            
            logger.info(f"正在初始化向量数据库: {db_type}")
            
            if db_type == "chroma":
                # ChromaDB初始化逻辑
                self.chroma_client = "chroma-client-mock"
                logger.info("ChromaDB 客户端初始化完成")
            else:
                logger.warning(f"不支持的数据库类型: {db_type}")
            
            return True
            
        except Exception as e:
            logger.error(f"向量数据库初始化失败: {e}")
            return False
    
    def generate_embedding(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        try:
            # 模拟嵌入向量生成
            # 实际实现中需要调用实际的嵌入模型
            
            # 使用文本的hash作为伪随机种子生成一致的向量
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            np.random.seed(int(text_hash[:8], 16))
            
            dimension = self.config.get("embedding_model", {}).get("dimension", 1024)
            embedding = np.random.normal(0, 1, dimension).tolist()
            
            # 归一化向量
            embedding = np.array(embedding)
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"嵌入向量生成失败: {e}")
            return [0.0] * self.config.get("embedding_model", {}).get("dimension", 1024)
    
    def chunk_text_medical(self, text: str, metadata: Dict) -> List[Dict]:
        """医学文本智能分块"""
        try:
            chunks = []
            
            # 医学知识模式检测
            medical_patterns = self.medical_config.get("medical_patterns", [])
            chunk_size = self.medical_config.get("chunk_size", 1000)
            chunk_overlap = self.medical_config.get("chunk_overlap", 200)
            
            # 分割文本为段落
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            current_chunk = ""
            current_metadata = metadata.copy()
            chunk_index = 0
            
            for paragraph in paragraphs:
                # 检查是否应该开始新块
                if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                    # 完成当前块
                    chunk_data = {
                        "id": f"chunk_{chunk_index}",
                        "content": current_chunk.strip(),
                        "metadata": current_metadata.copy(),
                        "chunk_index": chunk_index
                    }
                    
                    # 生成嵌入向量
                    chunk_data["embedding"] = self.generate_embedding(current_chunk)
                    
                    chunks.append(chunk_data)
                    
                    # 为下一块准备
                    overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                    current_chunk = overlap_text + "\n\n" + paragraph
                    chunk_index += 1
                else:
                    # 添加到当前块
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
            
            # 处理最后一块
            if current_chunk.strip():
                chunk_data = {
                    "id": f"chunk_{chunk_index}",
                    "content": current_chunk.strip(),
                    "metadata": current_metadata.copy(),
                    "chunk_index": chunk_index
                }
                
                # 生成嵌入向量
                chunk_data["embedding"] = self.generate_embedding(current_chunk)
                
                chunks.append(chunk_data)
            
            logger.info(f"文本分块完成，生成 {len(chunks)} 个块")
            return chunks
            
        except Exception as e:
            logger.error(f"文本分块失败: {e}")
            return []
    
    def process_medical_documents(self, input_dir: str, output_dir: str) -> Dict:
        """批量处理医学文档"""
        try:
            logger.info(f"开始批量处理医学文档，目录: {input_dir}")
            
            # 检查输入目录
            if not os.path.exists(input_dir):
                logger.error(f"输入目录不存在: {input_dir}")
                return {"success": False, "error": f"目录不存在: {input_dir}"}
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 查找处理后的知识块文件
            processed_files = []
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.endswith('_chunks.json'):
                        processed_files.append(os.path.join(root, file))
            
            if not processed_files:
                logger.warning(f"在 {input_dir} 中未找到处理后的知识块文件")
                return {"success": False, "error": "未找到处理后的知识块文件"}
            
            # 初始化模型和数据库
            if not self.initialize_embedding_model():
                return {"success": False, "error": "嵌入模型初始化失败"}
            
            if not self.initialize_vector_database():
                return {"success": False, "error": "向量数据库初始化失败"}
            
            all_vectorized_chunks = []
            document_stats = {
                "total_documents": len(processed_files),
                "total_chunks": 0,
                "successful_documents": 0,
                "failed_documents": 0
            }
            
            # 处理每个文档
            for file_path in processed_files:
                try:
                    logger.info(f"处理文档: {file_path}")
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        chunks_data = json.load(f)
                    
                    # 为每个知识块生成嵌入向量
                    vectorized_chunks = []
                    for chunk in chunks_data:
                        # 生成嵌入向量
                        embedding = self.generate_embedding(chunk.get("content", ""))
                        
                        # 添加向量信息
                        vectorized_chunk = chunk.copy()
                        vectorized_chunk["embedding"] = embedding
                        vectorized_chunk["vectorized_at"] = datetime.now().isoformat()
                        
                        vectorized_chunks.append(vectorized_chunk)
                        all_vectorized_chunks.append(vectorized_chunk)
                    
                    document_stats["total_chunks"] += len(vectorized_chunks)
                    document_stats["successful_documents"] += 1
                    
                    logger.info(f"文档 {file_path} 处理完成，生成 {len(vectorized_chunks)} 个向量化块")
                    
                except Exception as e:
                    logger.error(f"处理文档 {file_path} 失败: {e}")
                    document_stats["failed_documents"] += 1
                    continue
            
            # 保存所有向量化块
            output_file = os.path.join(output_dir, "vectorized_chunks.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_vectorized_chunks, f, ensure_ascii=False, indent=2)
            
            # 保存处理统计
            stats_file = os.path.join(output_dir, "vectorization_stats.json")
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(document_stats, f, ensure_ascii=False, indent=2)
            
            logger.info(f"向量化处理完成，结果保存到: {output_file}")
            logger.info(f"处理统计: {document_stats}")
            
            return {
                "success": True,
                "output_file": output_file,
                "stats": document_stats,
                "total_chunks": len(all_vectorized_chunks)
            }
            
        except Exception as e:
            logger.error(f"批量处理失败: {e}")
            return {"success": False, "error": str(e)}
    
    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """检索相似知识块"""
        try:
            # 生成查询嵌入向量
            query_embedding = self.generate_embedding(query)
            
            # 加载向量化块数据
            vectorized_file = "data/vectorized/vectorized_chunks.json"
            if not os.path.exists(vectorized_file):
                logger.error(f"向量化文件不存在: {vectorized_file}")
                return []
            
            with open(vectorized_file, 'r', encoding='utf-8') as f:
                all_chunks = json.load(f)
            
            # 计算相似度
            similarities = []
            for chunk in all_chunks:
                if "embedding" in chunk:
                    chunk_embedding = np.array(chunk["embedding"])
                    query_vec = np.array(query_embedding)
                    
                    # 余弦相似度
                    similarity = np.dot(chunk_embedding, query_vec) / (
                        np.linalg.norm(chunk_embedding) * np.linalg.norm(query_vec)
                    )
                    
                    similarities.append({
                        "chunk": chunk,
                        "similarity": float(similarity)
                    })
            
            # 按相似度排序
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # 返回Top-K结果
            top_results = similarities[:top_k]
            
            logger.info(f"检索查询: '{query}', 返回 {len(top_results)} 个结果")
            
            return top_results
            
        except Exception as e:
            logger.error(f"相似度检索失败: {e}")
            return []


def main():
    """主函数 - 向量化处理示例"""
    print("=== 医学知识向量化引擎 ===")
    
    # 初始化向量引擎
    engine = MedicalVectorEngine()
    
    # 批量处理医学文档
    input_directory = "data"
    output_directory = "data/vectorized"
    
    result = engine.process_medical_documents(input_directory, output_directory)
    
    if result["success"]:
        print(f"✅ 向量化处理成功!")
        print(f"📊 处理统计: {result['stats']}")
        print(f"💾 输出文件: {result['output_file']}")
        print(f"🔢 总知识块数: {result['total_chunks']}")
        
        # 测试检索功能
        print("\n=== 测试检索功能 ===")
        test_query = "心脏病的病因和病理"
        results = engine.search_similar_chunks(test_query, top_k=3)
        
        print(f"查询: {test_query}")
        print("检索结果:")
        for i, result_item in enumerate(results, 1):
            chunk = result_item["chunk"]
            similarity = result_item["similarity"]
            print(f"{i}. 相似度: {similarity:.4f}")
            print(f"   内容: {chunk['content'][:100]}...")
            print(f"   来源: {chunk.get('metadata', {}).get('document_title', 'N/A')}")
            print()
    else:
        print(f"❌ 向量化处理失败: {result['error']}")


if __name__ == "__main__":
    main()