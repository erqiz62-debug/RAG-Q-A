#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精确溯源医学问答系统 - 本地版本
完全脱离RagFlow，使用本地向量数据库进行检索
支持到页码和句子级别的溯源
"""

import sys
import os
import json
import re
import logging
import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict, Counter
import jieba
import jieba.posseg as pseg
import hashlib

# 添加scripts目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from local_vector_database import LocalVectorDatabase

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PreciseCitationMedicalQASystemLocal:
    """精确溯源医学问答系统 - 本地版本"""
    
    def __init__(self, 
                 vector_db_dir: str = "./data/chroma_db",
                 collection_name: str = "medical_knowledge"):
        """
        初始化精确溯源医学问答系统
        
        Args:
            vector_db_dir: 向量数据库目录
            collection_name: 集合名称
        """
        self.vector_db_dir = vector_db_dir
        self.collection_name = collection_name
        
        # 初始化向量数据库
        self.vector_db = LocalVectorDatabase(vector_db_dir, collection_name)
        
        # 医学同义词词典
        self.medical_synonyms = self._load_synonyms()
        self.query_expansion_keywords = self._load_query_expansion_keywords()
        
        # 嵌入模型（用于生成查询向量）
        self.embedding_model = None
        self._load_embedding_model()
        
        # 嵌入缓存
        self.embedding_cache = {}
        
        logger.info("精确溯源医学问答系统（本地版本）初始化完成")
    
    def _load_embedding_model(self):
        """加载嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info("加载嵌入模型: BAAI/bge-large-zh")
            self.embedding_model = SentenceTransformer('BAAI/bge-large-zh')
            logger.info("嵌入模型加载成功")
            
        except ImportError:
            logger.warning("sentence-transformers未安装，将使用简化向量生成")
            self.embedding_model = None
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {e}")
            self.embedding_model = None
    
    def _generate_query_embedding(self, query: str) -> List[float]:
        """
        生成查询向量
        
        Args:
            query: 查询文本
            
        Returns:
            查询向量
        """
        if self.embedding_model is not None:
            try:
                embedding = self.embedding_model.encode(query)
                return embedding.tolist()
            except Exception as e:
                logger.error(f"生成查询向量失败: {e}")
        
        # 备用方案：使用hash生成向量
        return self._generate_hash_embedding(query)
    
    def _generate_hash_embedding(self, text: str) -> List[float]:
        """
        使用hash生成向量（备用方案）
        
        Args:
            text: 文本内容
            
        Returns:
            向量列表
        """
        hash_obj = hashlib.md5(text.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()
        
        # 转换为1024维向量
        vector = []
        for i in range(0, len(hash_hex), 8):
            chunk = hash_hex[i:i+8]
            value = int(chunk, 16) / (16**8 - 1)
            vector.extend([value] * 256)  # 扩展到1024维
        
        return vector[:1024]  # 确保是1024维
    
    def process_query_with_precise_citations(self, query: str, 
                                          top_k: int = 5,
                                          vector_weight: float = 0.7,
                                          keyword_weight: float = 0.3) -> Dict[str, Any]:
        """
        处理查询并返回精确溯源的答案
        
        Args:
            query: 用户查询
            top_k: 返回结果数量
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
            
        Returns:
            包含答案和精确溯源信息的字典
        """
        logger.info(f"处理查询: {query}")
        
        # 生成查询向量
        query_embedding = self._generate_query_embedding(query)
        
        # 混合检索
        search_results = self.vector_db.hybrid_search(
            query_embedding=query_embedding,
            query=query,
            top_k=top_k,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight
        )
        
        if not search_results:
            return {
                'query': query,
                'answer': '抱歉，我在当前知识库中没有找到与您的问题相关的信息。',
                'confidence': 0.0,
                'precise_citations': [],
                'precise_citation_stats': {},
                'related_questions': []
            }
        
        # 构建答案
        answer_parts = []
        citations = []
        
        for result in search_results:
            chunk_id = result.get('chunk_id')
            chunk_data = self.vector_db.get_chunk_content(chunk_id)
            
            if chunk_data:
                content = chunk_data.get('content', '')
                metadata = chunk_data.get('metadata', {})
                
                # 添加到答案
                answer_parts.append(content)
                
                # 构建精确溯源信息
                citation = {
                    'source_document': metadata.get('_document_name', 'Unknown'),
                    'chapter': metadata.get('chapter', 'N/A'),
                    'section': metadata.get('section', 'N/A'),
                    'page_range': self._format_page_range(metadata.get('page', 'N/A')),
                    'line_range': self._format_line_range(metadata.get('line_start', 'N/A'), 
                                                         metadata.get('line_end', 'N/A')),
                    'relevance_score': result.get('fused_score', 0.5),
                    'completeness_score': self._calculate_completeness_score(content),
                    'sentence_citations': self._extract_sentence_citations(content, query)
                }
                citations.append(citation)
        
        # 合并答案
        answer = self._merge_answer_parts(answer_parts)
        
        # 计算置信度
        confidence = self._calculate_confidence(search_results)
        
        # 生成相关问题
        related_questions = self._generate_related_questions(query, search_results)
        
        # 计算统计信息
        stats = self._calculate_citation_stats(citations)
        
        return {
            'query': query,
            'answer': answer,
            'confidence': confidence,
            'precise_citations': citations,
            'precise_citation_stats': stats,
            'related_questions': related_questions
        }
    
    def _format_page_range(self, page: Any) -> str:
        """格式化页码范围"""
        if isinstance(page, (list, tuple)):
            if len(page) == 1:
                return f"第{page[0]}页"
            elif len(page) > 1:
                return f"第{min(page)}-{max(page)}页"
        elif page and page != 'N/A':
            return f"第{page}页"
        return "N/A"
    
    def _format_line_range(self, line_start: Any, line_end: Any) -> str:
        """格式化行号范围"""
        if line_start and line_end and line_start != 'N/A' and line_end != 'N/A':
            return f"第{line_start}-{line_end}行"
        elif line_start and line_start != 'N/A':
            return f"第{line_start}行"
        return "N/A"
    
    def _calculate_completeness_score(self, content: str) -> float:
        """计算内容完整性分数"""
        if not content:
            return 0.0
        
        # 基于内容长度和结构计算完整性
        length_score = min(len(content) / 500, 1.0)  # 假设500字符为完整内容
        
        # 检查是否包含关键结构元素
        structure_score = 0.0
        if '：' in content or ':' in content:
            structure_score += 0.3
        if '、' in content or '，' in content:
            structure_score += 0.3
        if '。' in content:
            structure_score += 0.4
        
        return (length_score + structure_score) / 2
    
    def _extract_sentence_citations(self, content: str, query: str) -> List[Dict[str, Any]]:
        """提取句子级引用"""
        sentences = re.split(r'[。！？]', content)
        sentence_citations = []
        
        query_keywords = set(jieba.cut(query))
        
        for i, sentence in enumerate(sentences, 1):
            if not sentence.strip():
                continue
            
            # 计算句子相关性
            sentence_words = set(jieba.cut(sentence))
            overlap = query_keywords.intersection(sentence_words)
            
            if overlap:
                relevance_score = len(overlap) / len(query_keywords)
                if relevance_score > 0.1:  # 相关性阈值
                    sentence_citations.append({
                        'sentence_number': i,
                        'sentence_text': sentence.strip(),
                        'relevance_score': relevance_score,
                        'matched_keywords': list(overlap)
                    })
        
        # 按相关性排序
        sentence_citations.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return sentence_citations[:5]  # 返回最相关的5个句子
    
    def _merge_answer_parts(self, parts: List[str]) -> str:
        """合并答案部分"""
        if not parts:
            return "抱歉，没有找到相关信息。"
        
        # 去重和合并
        unique_parts = []
        seen = set()
        
        for part in parts:
            part_hash = hashlib.md5(part.encode('utf-8')).hexdigest()
            if part_hash not in seen:
                seen.add(part_hash)
                unique_parts.append(part)
        
        # 合并为连贯的答案
        answer = "\n\n".join(unique_parts)
        
        # 添加总结性语句
        if len(unique_parts) > 1:
            answer = "根据医学知识库，相关信息如下：\n\n" + answer
        
        return answer
    
    def _calculate_confidence(self, search_results: List[Dict[str, Any]]) -> float:
        """计算置信度"""
        if not search_results:
            return 0.0
        
        # 基于融合分数计算置信度
        fused_scores = [r.get('fused_score', 0) for r in search_results]
        avg_score = sum(fused_scores) / len(fused_scores)
        
        # 考虑结果数量
        result_count_factor = min(len(search_results) / 3, 1.0)
        
        return avg_score * result_count_factor
    
    def _generate_related_questions(self, query: str, 
                                   search_results: List[Dict[str, Any]]) -> List[str]:
        """生成相关问题"""
        related_questions = []
        
        # 基于查询类型生成相关问题
        query_type = self._classify_query_type(query)
        
        if query_type == 'what':
            related_questions.extend([
                f"{query}的病因是什么？",
                f"{query}如何诊断？",
                f"{query}的治疗方法有哪些？"
            ])
        elif query_type == 'how':
            related_questions.extend([
                f"{query}的原理是什么？",
                f"{query}有哪些注意事项？",
                f"{query}的效果如何？"
            ])
        elif query_type == 'why':
            related_questions.extend([
                f"{query}如何预防？",
                f"{query}有哪些风险因素？",
                f"{query}的机制是什么？"
            ])
        
        # 基于检索结果生成相关问题
        for result in search_results[:2]:
            chunk_id = result.get('chunk_id')
            chunk_data = self.vector_db.get_chunk_content(chunk_id)
            if chunk_data:
                content = chunk_data.get('content', '')
                keywords = chunk_data.get('keywords', [])
                
                for keyword in keywords[:2]:
                    if keyword not in query:
                        related_questions.append(f"{keyword}与{query}有什么关系？")
        
        # 去重并限制数量
        unique_questions = list(set(related_questions))
        return unique_questions[:5]
    
    def _classify_query_type(self, query: str) -> str:
        """分类查询类型"""
        query_lower = query.lower().strip()
        
        if any(word in query_lower for word in ['什么是', '什么叫做', '定义', '概念', '意思', '是什么']):
            return 'what'
        elif any(word in query_lower for word in ['如何', '怎么', '怎样', '方式', '方法', '进行']):
            return 'how'
        elif any(word in query_lower for word in ['为什么', '原因', '为何', '缘故', '怎么会']):
            return 'why'
        elif any(word in query_lower for word in ['哪些', '什么', '种类', '类型', '分类', '包括']):
            return 'which'
        else:
            return 'general'
    
    def _calculate_citation_stats(self, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算引用统计信息"""
        if not citations:
            return {}
        
        unique_sources = set(c['source_document'] for c in citations)
        all_pages = []
        all_lines = []
        
        for citation in citations:
            # 解析页码
            page_range = citation.get('page_range', 'N/A')
            if '第' in page_range and '页' in page_range:
                page_str = page_range.replace('第', '').replace('页', '')
                if '-' in page_str:
                    start, end = page_str.split('-')
                    all_pages.extend(range(int(start), int(end) + 1))
                else:
                    all_pages.append(int(page_str))
            
            # 解析行号
            line_range = citation.get('line_range', 'N/A')
            if '第' in line_range and '行' in line_range:
                line_str = line_range.replace('第', '').replace('行', '')
                if '-' in line_str:
                    start, end = line_str.split('-')
                    all_lines.extend(range(int(start), int(end) + 1))
                else:
                    all_lines.append(int(line_str))
        
        avg_relevance = sum(c['relevance_score'] for c in citations) / len(citations)
        
        return {
            'total_citations': len(citations),
            'unique_sources': len(unique_sources),
            'page_range': f"第{min(all_pages)}-{max(all_pages)}页" if all_pages else "N/A",
            'line_range': f"第{min(all_lines)}-{max(all_lines)}行" if all_lines else "N/A",
            'avg_relevance_score': avg_relevance
        }
    
    def _load_synonyms(self) -> Dict:
        """加载医学同义词词典"""
        return {
            '心脏病': ['心脏疾病', '心血管疾病', '心疾病', '心系疾病', '心血管问题', '心脏病'],
            '冠心病': ['冠状动脉粥样硬化', '冠状动脉疾病', '冠心病', '冠状动脉硬化', '冠心', '冠心病'],
            '心肌梗死': ['心肌梗死', '心梗', '急性心肌梗死', 'AMI', '心机梗死'],
            '心力衰竭': ['心衰', '心功能不全', '心力衰竭', '心脏衰竭', 'CHF'],
            '高血压': ['高血压病', '动脉高血压', '血压升高', 'HTN', '血压高'],
            '心律失常': ['心律不齐', '心律紊乱', '心律失常', '心律异常', '心律问题'],
            '胸痛': ['胸痛', '心前区疼痛', '胸部疼痛', '胸骨后疼痛', '胸口疼'],
            '气短': ['呼吸困难', '气短', '气促', '呼吸急促', '胸闷气短'],
            '心悸': ['心跳加速', '心悸', '心慌', '心率快', '心跳快'],
            '水肿': ['水肿', '浮肿', '肿胀', '水潴留', '肿'],
            '头晕': ['头晕', '眩晕', '头昏', '眩晕症', '头昏眼花'],
            '心电图': ['ECG', '心电图', '心电描记', '心电检查', '电图'],
            '超声心动图': ['超声心动图', '心脏彩超', 'UCG', '心脏超声', '彩超'],
            '治疗': ['治疗', '处理', '治疗方案', '疗法', '医疗措施', '治疗办法'],
            '诊断': ['诊断', '检查', '确诊', '识别', '判断', '诊断方法'],
            '预防': ['预防', '预防措施', '预防方法', '预防策略', '防护', '预防措施'],
            '药物': ['药物', '药品', '药剂', '药品治疗', '吃药', '用药'],
            '症状': ['症状', '表现', '征象', '现象', '临床表现', '体征'],
            '病因': ['病因', '原因', '致病因素', '诱因', '发病原因', '为什么'],
            '检查': ['检查', '检验', '诊断检查', '实验室检查', '体检'],
            '手术': ['手术', '外科手术', '介入治疗', '手术方法', '开刀'],
            '护理': ['护理措施', '护理方法', '护理要点', '护理程序', '照护']
        }
    
    def _load_query_expansion_keywords(self) -> Dict:
        """加载查询扩展关键词"""
        return {
            '症状': ['表现', '征象', '现象', '特征', '体征', '临床表现', '反应', '感受'],
            '诊断': ['检查', '确诊', '识别', '判断', '诊断标准', '鉴别诊断', '检查方法'],
            '治疗': ['处理', '治疗方案', '疗法', '药物', '手术', '治疗原则', '治疗方法'],
            '病因': ['原因', '致病因素', '诱因', '起源', '发病机制', '病理', '为什么'],
            '预防': ['预防措施', '预防方法', '预防策略', '防护', '预防性治疗', '避免'],
            '预后': ['结局', '转归', '后果', '结果', '预后评估', '康复', '恢复'],
            '药物': ['药品', '药剂', '药物治疗', '给药', '剂量', '用法', '吃药'],
            '检查': ['检验', '诊断检查', '实验室检查', '影像学检查', '辅助检查', '体检'],
            '手术': ['外科手术', '介入治疗', '手术方法', '手术指征', '手术方式', '开刀'],
            '护理': ['护理措施', '护理方法', '护理要点', '护理程序', '照护', '照顾']
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        vector_db_stats = self.vector_db.get_collection_stats()
        
        return {
            "系统状态": "正常运行",
            "系统类型": "精确溯源医学问答系统（本地版本）",
            "Python版本": "3.7+ 兼容",
            "向量数据库": "ChromaDB/内存存储",
            "知识库条目": vector_db_stats['total_chunks'],
            "集合名称": vector_db_stats['collection_name'],
            "倒排索引大小": vector_db_stats['inverted_index_size'],
            "嵌入模型": "BAAI/bge-large-zh" if self.embedding_model else "Hash向量",
            "最后更新": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


def main():
    """主函数 - 演示精确溯源功能"""
    qa_system = PreciseCitationMedicalQASystemLocal()
    
    # 测试查询
    test_queries = [
        "什么是心脏病？",
        "冠心病的症状有哪些？",
        "如何治疗高血压？",
        "心力衰竭的原因是什么？",
        "心绞痛的典型表现"
    ]
    
    print("=" * 100)
    print("精确溯源医学问答系统演示（本地版本）")
    print("=" * 100)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n【查询 {i}】: {query}")
        print("=" * 80)
        
        response = qa_system.process_query_with_precise_citations(query)
        
        print(f"📝 回答: {response['answer']}")
        print(f"🎯 置信度: {response['confidence']:.3f}")
        print(f"📊 引用数量: {len(response['precise_citations'])}")
        
        # 显示精确溯源信息
        if response['precise_citations']:
            print("\n📚 精确溯源信息:")
            for j, citation in enumerate(response['precise_citations'], 1):
                print(f"  引用 {j}:")
                print(f"    📄 源文档: {citation['source_document']}")
                print(f"    📑 页码: {citation['page_range']}")
                print(f"    📝 行号: {citation['line_range']}")
                print(f"    🎯 相关性: {citation['relevance_score']:.3f}")
                print(f"    📊 完整性: {citation['completeness_score']:.3f}")
                
                # 显示句子级引用
                if citation['sentence_citations']:
                    print(f"    📍 关键句子:")
                    for sentence in citation['sentence_citations'][:2]:
                        print(f"      • {sentence['sentence_text'][:100]}...")
        
        # 显示统计信息
        stats = response['precise_citation_stats']
        print(f"\n📈 统计信息:")
        print(f"  • 总引用数: {stats['total_citations']}")
        print(f"  • 独特源文件: {stats['unique_sources']}")
        print(f"  • 页码覆盖: {stats['page_range']}")
        print(f"  • 行号覆盖: {stats['line_range']}")
        print(f"  • 平均相关性: {stats['avg_relevance_score']:.3f}")
        
        # 显示相关问题
        if response['related_questions']:
            print(f"\n💡 相关问题建议:")
            for related_q in response['related_questions'][:3]:
                print(f"  • {related_q}")
        
        print("\n" + "-" * 80)


if __name__ == "__main__":
    main()
