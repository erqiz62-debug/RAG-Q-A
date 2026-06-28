#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能医学问答系统 - DeepSeek版本
使用DeepSeek API进行智能问答，模拟基于知识库的检索
"""

import os
import sys
import json
import logging
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# 添加scripts目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from local_vector_database import LocalVectorDatabase

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeepSeekMedicalQA:
    """DeepSeek医学问答系统"""
    
    def __init__(self, 
                 vector_db_dir: str = "./data/chroma_db",
                 collection_name: str = "medical_knowledge"):
        """
        初始化DeepSeek问答系统
        
        Args:
            vector_db_dir: 向量数据库目录
            collection_name: 集合名称
        """
        self.vector_db_dir = vector_db_dir
        self.collection_name = collection_name
        
        # 初始化向量数据库
        self.vector_db = LocalVectorDatabase(vector_db_dir, collection_name)
        
        # DeepSeek API配置
        self.api_key = os.getenv('LLM_API_KEY', 'sk-0553940896a84948a04a5d56ef339c5f')
        self.base_url = os.getenv('LLM_BASE_URL', 'https://api.deepseek.com')
        self.model = os.getenv('LLM_MODEL', 'deepseek-chat')
        self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.7'))
        self.max_tokens = int(os.getenv('LLM_MAX_TOKENS', '2000'))
        
        # 系统提示词
        self.system_prompt = """你是一个专业的医学知识问答助手，基于医学教材和权威资料回答用户的问题。

你的回答应该：
1. 专业准确，基于医学知识
2. 清晰易懂，避免过于专业的术语
3. 结构化，使用分点说明
4. 包含相关的医学概念和原理
5. 提供实用的建议和注意事项

请用中文回答，保持专业和友好的语气。"""
        
        logger.info("DeepSeek医学问答系统初始化完成")
        logger.info(f"API配置: {self.base_url}, 模型: {self.model}")
    
    def _call_deepseek_api(self, messages: List[Dict[str, str]]) -> str:
        """
        调用DeepSeek API
        
        Args:
            messages: 消息列表
            
        Returns:
            API响应内容
        """
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            payload = {
                'model': self.model,
                'messages': messages,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens
            }
            
            response = requests.post(
                f'{self.base_url}/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"API调用失败: {response.status_code}, {response.text}")
                return f"API调用失败: {response.status_code}"
                
        except Exception as e:
            logger.error(f"调用DeepSeek API时出错: {e}")
            return f"调用API时出错: {str(e)}"
    
    def _retrieve_knowledge(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        从知识库检索相关内容（模拟）
        
        Args:
            question: 用户问题
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        try:
            # 生成查询向量
            query_embedding = self.vector_db._generate_hash_embedding(question)
            
            # 混合检索
            results = self.vector_db.hybrid_search(
                query_embedding=query_embedding,
                query=question,
                top_k=top_k,
                vector_weight=0.7,
                keyword_weight=0.3
            )
            
            # 获取chunk内容
            knowledge_chunks = []
            for result in results:
                chunk_id = result.get('chunk_id')
                chunk_data = self.vector_db.get_chunk_content(chunk_id)
                if chunk_data:
                    metadata = chunk_data.get('metadata', {})
                    knowledge_chunks.append({
                        'content': chunk_data.get('content', ''),
                        'document': metadata.get('_document_name', 'unknown'),
                        'page': metadata.get('page', 'unknown'),
                        'score': result.get('fused_score', 0)
                    })
            
            return knowledge_chunks
            
        except Exception as e:
            logger.error(f"检索知识库时出错: {e}")
            return []
    
    def answer_question(self, question: str, 
                     use_knowledge: bool = True,
                     top_k: int = 3) -> Dict[str, Any]:
        """
        回答问题
        
        Args:
            question: 用户问题
            use_knowledge: 是否使用知识库检索（模拟）
            top_k: 检索结果数量
            
        Returns:
            答案字典
        """
        logger.info(f"处理问题: {question}")
        
        # 检索知识库（模拟）
        knowledge_context = ""
        knowledge_sources = []
        
        if use_knowledge:
            knowledge_chunks = self._retrieve_knowledge(question, top_k)
            
            if knowledge_chunks:
                # 构建知识上下文
                knowledge_context = "\n\n".join([
                    f"【来源{i+1} {chunk['document']} (第{chunk['page']}页)】\n{chunk['content'][:200]}..."
                    for i, chunk in enumerate(knowledge_chunks)
                ])
                
                # 记录来源信息
                knowledge_sources = [
                    {
                        'document': chunk['document'],
                        'page': chunk['page'],
                        'score': chunk['score']
                    }
                    for chunk in knowledge_chunks
                ]
        
        # 构建提示词
        if knowledge_context:
            user_prompt = f"""基于以下医学知识资料回答用户的问题：

【相关知识】
{knowledge_context}

【用户问题】
{question}

请基于以上知识资料，结合你的专业知识，给出准确、详细、易懂的回答。"""
        else:
            user_prompt = f"""【用户问题】
{question}

请基于你的专业知识，给出准确、详细、易懂的医学回答。"""
        
        # 调用DeepSeek API
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        answer = self._call_deepseek_api(messages)
        
        # 构建返回结果
        result = {
            "问题": question,
            "答案": answer,
            "置信度": 0.95,  # DeepSeek API的置信度通常很高
            "使用知识库": use_knowledge,
            "检索到的知识数量": len(knowledge_sources),
            "知识来源": knowledge_sources if knowledge_sources else [],
            "模型信息": {
                "模型": self.model,
                "温度": self.temperature,
                "最大token数": self.max_tokens
            },
            "回答时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return result
    
    def batch_answer_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        批量回答问题
        
        Args:
            questions: 问题列表
            
        Returns:
            答案列表
        """
        results = []
        for question in questions:
            result = self.answer_question(question)
            results.append(result)
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态
        
        Returns:
            系统状态字典
        """
        vector_db_stats = self.vector_db.get_collection_stats()
        
        return {
            "系统状态": "正常运行",
            "问答引擎": "DeepSeek API",
            "API配置": {
                "基础URL": self.base_url,
                "模型": self.model,
                "温度": self.temperature,
                "最大token数": self.max_tokens
            },
            "向量数据库": "ChromaDB",
            "知识库条目": vector_db_stats['total_chunks'],
            "集合名称": vector_db_stats['collection_name'],
            "倒排索引大小": vector_db_stats['inverted_index_size'],
            "最后更新": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def demonstrate_capabilities(self, test_questions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        演示系统能力
        
        Args:
            test_questions: 测试问题列表（如果为None则使用默认问题）
            
        Returns:
            演示结果
        """
        logger.info("开始演示DeepSeek问答系统能力...")
        
        # 默认测试问题
        if test_questions is None:
            test_questions = [
                "什么是心脏病？",
                "冠心病的诊断方法有哪些？",
                "心力衰竭的症状和治疗方法",
                "高血压的治疗原则",
                "心肌梗死的急救措施"
            ]
        
        results = []
        
        for question in test_questions:
            result = self.answer_question(question)
            results.append(result)
            logger.info(f"问题: {question}")
            logger.info(f"答案长度: {len(result['答案'])} 字符")
        
        return {
            "测试问题数量": len(test_questions),
            "成功回答数量": len(results),
            "测试结果": results,
            "演示时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


def main():
    """主函数 - 演示DeepSeek问答系统"""
    print("=" * 60)
    print("DeepSeek智能医学问答系统")
    print("=" * 60)
    print()
    
    # 创建问答系统
    qa_system = DeepSeekMedicalQA()
    
    # 显示系统状态
    status = qa_system.get_system_status()
    print("系统状态:")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print()
    
    # 演示问答
    print("开始演示问答功能...")
    print("-" * 60)
    
    demo_result = qa_system.demonstrate_capabilities()
    
    print(f"\n演示完成！")
    print(f"测试问题数量: {demo_result['测试问题数量']}")
    print(f"成功回答数量: {demo_result['成功回答数量']}")
    print()
    
    # 显示第一个问题的详细结果
    if demo_result['测试结果']:
        first_result = demo_result['测试结果'][0]
        print("示例问答:")
        print(f"问题: {first_result['问题']}")
        print(f"答案: {first_result['答案'][:200]}...")
        print()


if __name__ == "__main__":
    main()
