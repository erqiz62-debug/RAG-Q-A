#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版医学问答系统 - 支持宽泛问题和精确溯源
基于增强版知识库的智能医学问答系统
"""

import os
import json
import logging
import re
from typing import List, Dict, Tuple, Optional
import jieba
import jieba.posseg as pseg
from collections import Counter
import numpy as np
from difflib import SequenceMatcher

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMedicalQASystem:
    def __init__(self, knowledge_base_path: str):
        """初始化增强版医学问答系统"""
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = None
        self.medical_synonyms = self._load_synonyms()
        self.query_expansion_keywords = self._load_query_expansion_keywords()
        
        # 加载知识库
        self.load_knowledge_base()
    
    def _load_synonyms(self) -> Dict:
        """加载医学同义词词典"""
        return {
            '心脏病': ['心脏疾病', '心血管疾病', '心疾病', '心系疾病'],
            '冠心病': ['冠状动脉粥样硬化', '冠状动脉疾病', '冠心病', '冠状动脉硬化'],
            '心肌梗死': ['心肌梗死', '心梗', '急性心肌梗死', 'AMI'],
            '心力衰竭': ['心衰', '心功能不全', '心力衰竭', '心脏衰竭'],
            '高血压': ['高血压病', '动脉高血压', '血压升高', 'HTN'],
            '心律失常': ['心律不齐', '心律紊乱', '心律失常', '心律异常'],
            '胸痛': ['胸痛', '心前区疼痛', '胸部疼痛', '胸骨后疼痛'],
            '气短': ['呼吸困难', '气短', '气促', '呼吸急促'],
            '心悸': ['心跳加速', '心悸', '心慌', '心率快'],
            '水肿': ['水肿', '浮肿', '肿胀', '水潴留'],
            '头晕': ['头晕', '眩晕', '头昏', '眩晕症'],
            '心电图': ['ECG', '心电图', '心电描记', '心电检查'],
            '超声心动图': ['超声心动图', '心脏彩超', 'UCG', '心脏超声'],
            '治疗': ['治疗', '处理', '治疗方案', '疗法', '医疗措施'],
            '诊断': ['诊断', '检查', '确诊', '识别', '判断'],
            '预防': ['预防', '预防措施', '预防方法', '预防策略'],
            '药物': ['药物', '药品', '药剂', '药品治疗'],
            '症状': ['症状', '表现', '征象', '现象', '临床表现'],
            '病因': ['病因', '原因', '致病因素', '诱因', '发病原因'],
            '检查': ['检查', '检验', '诊断检查', '实验室检查']
        }
    
    def _load_query_expansion_keywords(self) -> Dict:
        """加载查询扩展关键词"""
        return {
            '症状': ['表现', '征象', '现象', '特征', '体征', '临床表现'],
            '诊断': ['检查', '确诊', '识别', '判断', '诊断标准', '鉴别诊断'],
            '治疗': ['处理', '治疗方案', '疗法', '药物', '手术', '治疗原则'],
            '病因': ['原因', '致病因素', '诱因', '起源', '发病机制', '病理'],
            '预防': ['预防措施', '预防方法', '预防策略', '防护', '预防性治疗'],
            '预后': ['结局', '转归', '后果', '结果', '预后评估', '康复'],
            '药物': ['药品', '药剂', '药物治疗', '给药', '剂量', '用法'],
            '检查': ['检验', '诊断检查', '实验室检查', '影像学检查', '辅助检查'],
            '手术': ['外科手术', '介入治疗', '手术方法', '手术指征', '手术方式'],
            '护理': ['护理措施', '护理方法', '护理要点', '护理程序', '照护']
        }
    
    def load_knowledge_base(self):
        """加载增强版知识库"""
        try:
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
            logger.info(f"成功加载知识库: {len(self.knowledge_base.get('chunks', []))} 个知识块")
        except Exception as e:
            logger.error(f"加载知识库失败: {e}")
            self.knowledge_base = {'chunks': [], 'metadata': {}}
    
    def expand_query_for_broad_questions(self, query: str) -> List[str]:
        """为宽泛问题生成扩展查询"""
        expanded_queries = [query]  # 原始查询
        
        # 预处理查询
        query_lower = query.lower().strip()
        
        # 检测查询类型
        query_type = self._classify_query_type(query)
        
        # 基于查询类型生成扩展
        if query_type == 'what':
            # "什么是"类型的问题
            expanded_queries.extend(self._expand_what_queries(query))
        elif query_type == 'how':
            # "如何"类型的问题
            expanded_queries.extend(self._expand_how_queries(query))
        elif query_type == 'why':
            # "为什么"类型的问题
            expanded_queries.extend(self._expand_why_queries(query))
        elif query_type == 'which':
            # "哪些"类型的问题
            expanded_queries.extend(self._expand_which_queries(query))
        
        # 添加同义词扩展
        synonyms = self._get_synonyms_in_query(query)
        for synonym in synonyms:
            expanded_queries.append(synonym)
            expanded_queries.append(f"{query} {synonym}")
        
        return list(set(expanded_queries))  # 去重
    
    def _classify_query_type(self, query: str) -> str:
        """分类查询类型"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['什么是', '什么是', '什么叫做', '定义', '概念']):
            return 'what'
        elif any(word in query_lower for word in ['如何', '怎么', '怎样', '方式', '方法']):
            return 'how'
        elif any(word in query_lower for word in ['为什么', '原因', '为何', '缘故']):
            return 'why'
        elif any(word in query_lower for word in ['哪些', '什么', '种类', '类型', '分类']):
            return 'which'
        else:
            return 'general'
    
    def _expand_what_queries(self, query: str) -> List[str]:
        """扩展"什么是"类型查询"""
        expansions = []
        
        # 提取核心概念
        core_concept = self._extract_core_concept(query)
        
        if core_concept:
            expansions.extend([
                f"{core_concept}的定义",
                f"{core_concept}的概念",
                f"什么是{core_concept}",
                f"{core_concept}的含义",
                f"{core_concept}的基本概念"
            ])
        
        return expansions
    
    def _expand_how_queries(self, query: str) -> List[str]:
        """扩展"如何"类型查询"""
        expansions = []
        
        core_concept = self._extract_core_concept(query)
        if core_concept:
            expansions.extend([
                f"{core_concept}的治疗方法",
                f"如何诊断{core_concept}",
                f"{core_concept}的处理",
                f"{core_concept}的管理",
                f"{core_concept}的预防"
            ])
        
        return expansions
    
    def _expand_why_queries(self, query: str) -> List[str]:
        """扩展"为什么"类型查询"""
        expansions = []
        
        core_concept = self._extract_core_concept(query)
        if core_concept:
            expansions.extend([
                f"{core_concept}的病因",
                f"{core_concept}的发病机制",
                f"为什么会有{core_concept}",
                f"{core_concept}的形成原因",
                f"{core_concept}的病理生理"
            ])
        
        return expansions
    
    def _expand_which_queries(self, query: str) -> List[str]:
        """扩展"哪些"类型查询"""
        expansions = []
        
        core_concept = self._extract_core_concept(query)
        if core_concept:
            expansions.extend([
                f"{core_concept}的症状",
                f"{core_concept}的类型",
                f"{core_concept}的分类",
                f"{core_concept}的种类",
                f"{core_concept}包括什么"
            ])
        
        return expansions
    
    def _extract_core_concept(self, query: str) -> str:
        """提取查询中的核心概念"""
        # 移除疑问词
        query_cleaned = re.sub(r'[什么是如何为什么哪些怎么怎样]', '', query)
        
        # 使用jieba分词
        words = jieba.cut(query_cleaned.strip())
        
        # 寻找医学术语
        for word in words:
            if self._is_medical_term(word):
                return word
        
        return query_cleaned.strip()
    
    def _is_medical_term(self, word: str) -> bool:
        """检查是否是医学术语"""
        # 从知识库中获取所有医学术语
        all_medical_terms = set()
        for chunk in self.knowledge_base.get('chunks', []):
            all_medical_terms.update(chunk.get('medical_terms', []))
        
        # 检查是否在同义词中
        for synonyms in self.medical_synonyms.values():
            if word in synonyms:
                return True
        
        return word in all_medical_terms
    
    def _get_synonyms_in_query(self, query: str) -> List[str]:
        """获取查询中的同义词"""
        synonyms = []
        for standard_term, synonym_list in self.medical_synonyms.items():
            if standard_term in query:
                synonyms.extend(synonym_list)
            elif any(synonym in query for synonym in synonym_list):
                synonyms.append(standard_term)
                synonyms.extend([s for s in synonym_list if s != standard_term])
        
        return synonyms
    
    def intelligent_retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """智能检索 - 支持宽泛问题"""
        logger.info(f"处理查询: {query}")
        
        # 1. 生成扩展查询
        expanded_queries = self.expand_query_for_broad_questions(query)
        logger.info(f"生成了 {len(expanded_queries)} 个扩展查询")
        
        # 2. 对每个扩展查询进行检索
        all_candidates = []
        
        for expanded_query in expanded_queries:
            candidates = self._retrieve_for_query(expanded_query, top_k // len(expanded_queries))
            all_candidates.extend(candidates)
        
        # 3. 去重和排序
        unique_candidates = self._deduplicate_candidates(all_candidates)
        ranked_candidates = self._rank_candidates(unique_candidates, query)
        
        return ranked_candidates[:top_k]
    
    def _retrieve_for_query(self, query: str, top_k: int) -> List[Dict]:
        """针对单个查询进行检索"""
        chunks = self.knowledge_base.get('chunks', [])
        candidates = []
        
        query_words = set(jieba.cut(query))
        
        for chunk in chunks:
            score = self._calculate_chunk_relevance(chunk, query, query_words)
            if score > 0.1:  # 最低相关性阈值
                candidates.append({
                    'chunk': chunk,
                    'relevance_score': score,
                    'matched_query': query
                })
        
        # 按相关性排序
        candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
        return candidates[:top_k]
    
    def _calculate_chunk_relevance(self, chunk: Dict, query: str, query_words: set) -> float:
        """计算知识块相关性"""
        content = chunk.get('content', '')
        medical_terms = chunk.get('medical_terms', [])
        knowledge_types = chunk.get('knowledge_types', [])
        
        score = 0.0
        
        # 1. 医学术语匹配 (权重: 0.4)
        term_matches = len([term for term in medical_terms if term in query])
        if medical_terms:
            term_score = term_matches / len(medical_terms) * 0.4
            score += term_score
        
        # 2. 内容相似度 (权重: 0.3)
        content_words = set(jieba.cut(content))
        if content_words:
            overlap = len(query_words.intersection(content_words))
            similarity_score = overlap / max(len(query_words), len(content_words)) * 0.3
            score += similarity_score
        
        # 3. 知识类型匹配 (权重: 0.2)
        query_type = self._classify_query_type(query)
        if self._is_query_type_relevant(knowledge_types, query_type):
            score += 0.2
        
        # 4. 同义词匹配 (权重: 0.1)
        synonyms_in_query = self._get_synonyms_in_query(query)
        synonym_matches = len([syn for syn in synonyms_in_query if syn in content])
        if synonyms_in_query:
            synonym_score = synonym_matches / len(synonyms_in_query) * 0.1
            score += synonym_score
        
        return min(score, 1.0)
    
    def _is_query_type_relevant(self, knowledge_types: List[str], query_type: str) -> bool:
        """检查知识类型是否与查询类型相关"""
        type_mapping = {
            'what': ['诊断学', '解剖学', '病理学', '症状学'],
            'how': ['治疗学', '药物学', '护理学'],
            'why': ['病因学', '病理学', '流行病学'],
            'which': ['症状学', '分类学', '药物学']
        }
        
        relevant_types = type_mapping.get(query_type, [])
        return any(kt in knowledge_types for kt in relevant_types)
    
    def _deduplicate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """去重候选结果"""
        seen_chunks = set()
        unique_candidates = []
        
        for candidate in candidates:
            chunk_id = candidate['chunk']['id']
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _rank_candidates(self, candidates: List[Dict], original_query: str) -> List[Dict]:
        """重新排序候选结果"""
        # 基于原始查询的相关性重新评分
        query_words = set(jieba.cut(original_query))
        
        for candidate in candidates:
            chunk = candidate['chunk']
            # 重新计算相关性分数
            relevance = self._calculate_chunk_relevance(chunk, original_query, query_words)
            # 考虑完整性分数
            completeness = chunk.get('completeness_score', 0.5)
            # 考虑难度级别
            difficulty_bonus = self._get_difficulty_bonus(chunk)
            
            final_score = relevance * 0.7 + completeness * 0.2 + difficulty_bonus * 0.1
            candidate['final_score'] = final_score
        
        return sorted(candidates, key=lambda x: x['final_score'], reverse=True)
    
    def _get_difficulty_bonus(self, chunk: Dict) -> float:
        """根据难度级别给予奖励分数"""
        difficulty = chunk.get('difficulty_level', 'intermediate')
        bonus_map = {
            'basic': 0.1,
            'intermediate': 0.2,
            'advanced': 0.3
        }
        return bonus_map.get(difficulty, 0.1)
    
    def generate_response_with_citations(self, query: str, candidates: List[Dict]) -> Dict:
        """生成带引用的回答"""
        if not candidates:
            return {
                'answer': '抱歉，我在知识库中没有找到相关信息。',
                'confidence': 0.0,
                'citations': [],
                'related_questions': []
            }
        
        # 选择最佳候选结果
        best_candidates = candidates[:3]  # 取前3个最相关的结果
        
        # 生成综合回答
        answer_parts = []
        citations = []
        
        for i, candidate in enumerate(best_candidates, 1):
            chunk = candidate['chunk']
            content = chunk['content']
            
            # 生成回答片段
            if i == 1:
                answer_parts.append(f"根据医学资料：{content}")
            else:
                answer_parts.append(f"另外，{content}")
            
            # 生成精确引用
            citation = self.generate_precise_citation(chunk)
            citations.append(citation)
        
        # 计算整体置信度
        confidence = np.mean([c['final_score'] for c in best_candidates])
        
        # 生成相关问题
        related_questions = self._generate_related_questions(query, candidates)
        
        return {
            'answer': '\n\n'.join(answer_parts),
            'confidence': confidence,
            'citations': citations,
            'related_questions': related_questions,
            'query_analysis': {
                'original_query': query,
                'query_type': self._classify_query_type(query),
                'expanded_queries_count': len(self.expand_query_for_broad_questions(query))
            }
        }
    
    def generate_precise_citation(self, chunk: Dict) -> Dict:
        """生成精确的引用信息"""
        citation = {
            'source_file': chunk['source_file'],
            'page_number': chunk['page_number'],
            'block_index': chunk['block_index'],
            'knowledge_types': chunk.get('knowledge_types', []),
            'medical_terms': chunk.get('medical_terms', []),
            'relevance_score': chunk.get('relevance_score', 0)
        }
        
        # 添加句子级引用（如果有的话）
        if chunk.get('sentences'):
            # 选择最相关的句子
            most_relevant_sentence = chunk['sentences'][0]  # 简化处理
            citation['key_sentence'] = most_relevant_sentence['text']
            citation['sentence_position'] = most_relevant_sentence['bbox']
        
        return citation
    
    def _generate_related_questions(self, original_query: str, candidates: List[Dict]) -> List[str]:
        """生成相关问题建议"""
        related_questions = []
        
        query_type = self._classify_query_type(original_query)
        core_concept = self._extract_core_concept(original_query)
        
        if core_concept:
            # 基于查询类型生成相关问题
            if query_type == 'what':
                related_questions.extend([
                    f"如何诊断{core_concept}？",
                    f"如何治疗{core_concept}？",
                    f"{core_concept}的症状有哪些？"
                ])
            elif query_type == 'how':
                related_questions.extend([
                    f"什么是{core_concept}？",
                    f"{core_concept}的预防措施是什么？",
                    f"{core_concept}的预后如何？"
                ])
            elif query_type == 'why':
                related_questions.extend([
                    f"{core_concept}的症状有哪些？",
                    f"如何治疗{core_concept}？",
                    f"如何预防{core_concept}？"
                ])
        
        # 基于知识类型生成问题
        if candidates:
            knowledge_types = candidates[0]['chunk'].get('knowledge_types', [])
            if '症状学' in knowledge_types:
                related_questions.append(f"{core_concept}有哪些典型症状？")
            if '治疗学' in knowledge_types:
                related_questions.append(f"{core_concept}的治疗原则是什么？")
        
        return related_questions[:5]  # 返回最多5个相关问题
    
    def process_query(self, query: str, top_k: int = 10) -> Dict:
        """处理查询的主要方法"""
        logger.info(f"开始处理查询: {query}")
        
        # 1. 智能检索
        candidates = self.intelligent_retrieve(query, top_k)
        
        # 2. 生成带引用的回答
        response = self.generate_response_with_citations(query, candidates)
        
        # 3. 添加统计信息
        response['statistics'] = {
            'total_candidates_found': len(candidates),
            'knowledge_chunks_processed': len(self.knowledge_base.get('chunks', [])),
            'processing_timestamp': '2025-12-26'
        }
        
        logger.info(f"查询处理完成，置信度: {response['confidence']:.3f}")
        
        return response

def main():
    """测试函数"""
    qa_system = EnhancedMedicalQASystem(
        knowledge_base_path="d:\\BS\\medical-qa-ragflow\\data\\enhanced_medical_knowledge_base.json"
    )
    
    # 测试宽泛问题
    test_queries = [
        "什么是心脏病？",
        "如何治疗高血压？",
        "冠心病有哪些症状？",
        "为什么会出现心律失常？",
        "心力衰竭的诊断方法有哪些？"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"查询: {query}")
        print('='*50)
        
        response = qa_system.process_query(query)
        
        print(f"回答: {response['answer']}")
        print(f"置信度: {response['confidence']:.3f}")
        print(f"引用数量: {len(response['citations'])}")
        print(f"相关问题: {response['related_questions']}")

if __name__ == "__main__":
    main()