#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版医学问答系统 - 简化版
基于现有知识库，支持宽泛问题回答和精确溯源
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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplifiedEnhancedMedicalQASystem:
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
            '心脏病': ['心脏疾病', '心血管疾病', '心疾病', '心系疾病', '心血管问题'],
            '冠心病': ['冠状动脉粥样硬化', '冠状动脉疾病', '冠心病', '冠状动脉硬化', '冠心'],
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
    
    def load_knowledge_base(self):
        """加载知识库"""
        try:
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
            logger.info(f"成功加载知识库: {len(self.knowledge_base.get('chunks', []))} 个知识块")
        except Exception as e:
            logger.error(f"加载知识库失败: {e}")
            # 如果加载失败，创建空知识库
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
        
        # 生成更广泛的匹配查询
        broad_queries = self._generate_broad_queries(query)
        expanded_queries.extend(broad_queries)
        
        return list(set(expanded_queries))  # 去重
    
    def _generate_broad_queries(self, query: str) -> List[str]:
        """生成更广泛的匹配查询"""
        broad_queries = []
        
        # 提取核心概念
        core_concept = self._extract_core_concept(query)
        
        if core_concept:
            # 生成相关概念查询
            related_concepts = self._get_related_concepts(core_concept)
            for related in related_concepts:
                broad_queries.append(related)
                broad_queries.append(f"{related} 相关")
                broad_queries.append(f"关于 {related}")
        
        return broad_queries
    
    def _get_related_concepts(self, concept: str) -> List[str]:
        """获取相关概念"""
        # 基于同义词库查找相关概念
        related = []
        
        for standard_term, synonyms in self.medical_synonyms.items():
            if concept in synonyms or concept == standard_term:
                related.append(standard_term)
                related.extend(synonyms)
            elif any(syn in concept or concept in syn for syn in synonyms):
                related.append(standard_term)
                related.extend([s for s in synonyms if s != concept])
        
        return list(set(related))
    
    def _classify_query_type(self, query: str) -> str:
        """分类查询类型"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['什么是', '什么叫做', '定义', '概念', '意思']):
            return 'what'
        elif any(word in query_lower for word in ['如何', '怎么', '怎样', '方式', '方法', '进行']):
            return 'how'
        elif any(word in query_lower for word in ['为什么', '原因', '为何', '缘故', '怎么会']):
            return 'why'
        elif any(word in query_lower for word in ['哪些', '什么', '种类', '类型', '分类', '包括']):
            return 'which'
        else:
            return 'general'
    
    def _expand_what_queries(self, query: str) -> List[str]:
        """扩展"什么是"类型查询"""
        expansions = []
        
        core_concept = self._extract_core_concept(query)
        
        if core_concept:
            expansions.extend([
                f"{core_concept}的定义",
                f"{core_concept}的概念",
                f"什么是{core_concept}",
                f"{core_concept}的含义",
                f"{core_concept}的基本概念",
                f"{core_concept}是什么",
                f"关于{core_concept}",
                f"{core_concept}的介绍"
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
                f"{core_concept}的预防",
                f"怎么治疗{core_concept}",
                f"{core_concept}的治疗",
                f"治疗{core_concept}的方法"
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
                f"{core_concept}的病理生理",
                f"{core_concept}是怎么引起的",
                f"{core_concept}的致病原因"
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
                f"{core_concept}包括什么",
                f"{core_concept}有什么",
                f"{core_concept}的表现"
            ])
        
        return expansions
    
    def _extract_core_concept(self, query: str) -> str:
        """提取查询中的核心概念"""
        # 移除疑问词和修饰词
        query_cleaned = re.sub(r'[什么是如何为什么哪些怎么怎样怎么进行处理治疗诊断预防]', '', query)
        
        # 使用jieba分词
        words = jieba.cut(query_cleaned.strip())
        
        # 寻找医学术语
        for word in words:
            if self._is_medical_term(word):
                return word
        
        return query_cleaned.strip()
    
    def _is_medical_term(self, word: str) -> bool:
        """检查是否是医学术语"""
        # 检查是否在同义词中
        for synonyms in self.medical_synonyms.values():
            if word in synonyms:
                return True
        
        return word in self.medical_synonyms.keys()
    
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
            if score > 0.05:  # 降低最低相关性阈值
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
        metadata = chunk.get('metadata', {})
        
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
        
        # 3. 标题和章节匹配 (权重: 0.2)
        title_match = self._check_title_match(chunk, query)
        if title_match:
            score += 0.2
        
        # 4. 同义词匹配 (权重: 0.1)
        synonyms_in_query = self._get_synonyms_in_query(query)
        synonym_matches = len([syn for syn in synonyms_in_query if syn in content])
        if synonyms_in_query:
            synonym_score = synonym_matches / len(synonyms_in_query) * 0.1
            score += synonym_score
        
        return min(score, 1.0)
    
    def _check_title_match(self, chunk: Dict, query: str) -> bool:
        """检查标题匹配"""
        metadata = chunk.get('metadata', {})
        chapter = metadata.get('chapter', '')
        section = metadata.get('section', '')
        source_file = metadata.get('source_file', '')
        
        # 检查章节标题是否包含查询关键词
        title_parts = [chapter, section, source_file]
        for part in title_parts:
            if part and any(word in part for word in jieba.cut(query)):
                return True
        
        return False
    
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
            # 考虑章节匹配奖励
            chapter_bonus = self._get_chapter_match_bonus(chunk, original_query)
            
            final_score = relevance + chapter_bonus
            candidate['final_score'] = final_score
        
        return sorted(candidates, key=lambda x: x['final_score'], reverse=True)
    
    def _get_chapter_match_bonus(self, chunk: Dict, query: str) -> float:
        """根据章节匹配给予奖励"""
        metadata = chunk.get('metadata', {})
        chapter = metadata.get('chapter', '')
        
        if chapter:
            query_words = set(jieba.cut(query))
            chapter_words = set(jieba.cut(chapter))
            
            overlap = len(query_words.intersection(chapter_words))
            if overlap > 0:
                return min(overlap / len(query_words) * 0.2, 0.2)
        
        return 0.0
    
    def generate_response_with_citations(self, query: str, candidates: List[Dict]) -> Dict:
        """生成带引用的回答"""
        if not candidates:
            return {
                'answer': '抱歉，我在医学知识库中没有找到相关信息。您可以尝试使用更具体的医学术语提问，或者检查拼写是否正确。',
                'confidence': 0.0,
                'citations': [],
                'related_questions': self._generate_suggested_questions(query),
                'query_analysis': {
                    'original_query': query,
                    'query_type': self._classify_query_type(query),
                    'expanded_queries_count': len(self.expand_query_for_broad_questions(query)),
                    'suggestions': [
                        '尝试使用具体的医学术语',
                        '明确说明您想了解的方面（症状、治疗、诊断等）',
                        '检查术语拼写是否正确'
                    ]
                }
            }
        
        # 选择最佳候选结果
        best_candidates = candidates[:3]  # 取前3个最相关的结果
        
        # 生成综合回答
        answer_parts = []
        citations = []
        
        query_type = self._classify_query_type(query)
        
        for i, candidate in enumerate(best_candidates, 1):
            chunk = candidate['chunk']
            content = chunk['content']
            metadata = chunk.get('metadata', {})
            
            # 根据查询类型生成回答
            if query_type == 'what':
                answer_parts.append(f"根据医学资料，{content}")
            elif query_type == 'how':
                answer_parts.append(f"根据治疗指南，{content}")
            elif query_type == 'why':
                answer_parts.append(f"根据病因分析，{content}")
            elif query_type == 'which':
                answer_parts.append(f"根据临床资料，{content}")
            else:
                answer_parts.append(f"相关资料表明，{content}")
            
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
                'query_type': query_type,
                'expanded_queries_count': len(self.expand_query_for_broad_questions(query)),
                'total_candidates_found': len(candidates)
            }
        }
    
    def generate_precise_citation(self, chunk: Dict) -> Dict:
        """生成精确的引用信息"""
        metadata = chunk.get('metadata', {})
        
        citation = {
            'source_file': metadata.get('source_file', 'Unknown'),
            'page_number': metadata.get('page_number', 'N/A'),
            'chapter': metadata.get('chapter', 'N/A'),
            'section': metadata.get('section', 'N/A'),
            'knowledge_chain_id': chunk.get('id', 'N/A'),
            'content_preview': chunk.get('content', '')[:100] + '...' if len(chunk.get('content', '')) > 100 else chunk.get('content', ''),
            'medical_terms': chunk.get('medical_terms', []),
            'relevance_score': 0.0
        }
        
        # 如果有knowledge_chain信息
        if 'knowledge_chain' in chunk:
            chain = chunk['knowledge_chain']
            citation.update({
                'knowledge_chain_type': chain.get('type', 'N/A'),
                'chain_completeness': chain.get('completeness_score', 0.0),
                'chain_confidence': chain.get('confidence', 0.0)
            })
        
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
                    f"{core_concept}的症状有哪些？",
                    f"什么引起了{core_concept}？",
                    f"{core_concept}的预后如何？"
                ])
            elif query_type == 'how':
                related_questions.extend([
                    f"什么是{core_concept}？",
                    f"{core_concept}的症状有哪些？",
                    f"什么引起了{core_concept}？",
                    f"{core_concept}需要做什么检查？",
                    f"如何预防{core_concept}？"
                ])
            elif query_type == 'why':
                related_questions.extend([
                    f"{core_concept}的症状有哪些？",
                    f"如何治疗{core_concept}？",
                    f"如何预防{core_concept}？",
                    f"{core_concept}的诊断方法",
                    f"什么人群容易得{core_concept}？"
                ])
            elif query_type == 'which':
                related_questions.extend([
                    f"什么是{core_concept}？",
                    f"如何治疗{core_concept}？",
                    f"什么引起了{core_concept}？",
                    f"如何预防{core_concept}？",
                    f"{core_concept}的预后如何？"
                ])
        
        # 基于找到的医学术语生成问题
        if candidates:
            for candidate in candidates[:2]:  # 只从前两个候选中提取
                chunk = candidate['chunk']
                medical_terms = chunk.get('medical_terms', [])
                for term in medical_terms[:2]:  # 每个chunk最多取2个术语
                    if term != core_concept:
                        related_questions.append(f"什么是{term}？")
        
        return list(set(related_questions))[:5]  # 去重并限制数量
    
    def _generate_suggested_questions(self, original_query: str) -> List[str]:
        """为无结果查询生成建议问题"""
        suggestions = []
        
        core_concept = self._extract_core_concept(original_query)
        
        if core_concept:
            suggestions.extend([
                f"什么是{core_concept}？",
                f"如何治疗{core_concept}？",
                f"{core_concept}的症状有哪些？",
                f"什么引起了{core_concept}？",
                f"如何预防{core_concept}？"
            ])
        else:
            suggestions = [
                "心脏病有哪些症状？",
                "如何治疗高血压？",
                "什么是冠心病？",
                "心肌梗死的急救措施",
                "心律失常的诊断方法"
            ]
        
        return suggestions
    
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
            'processing_timestamp': '2025-12-26',
            'system_version': '2.0_enhanced_simplified'
        }
        
        logger.info(f"查询处理完成，置信度: {response['confidence']:.3f}")
        
        return response

def main():
    """测试函数"""
    qa_system = SimplifiedEnhancedMedicalQASystem(
        knowledge_base_path="d:\\BS\\medical-qa-ragflow\\data\\processed_chunks.json"
    )
    
    # 测试宽泛问题
    test_queries = [
        "什么是心脏病？",
        "如何治疗高血压？",
        "冠心病有哪些症状？",
        "为什么会出现心律失常？",
        "心力衰竭的诊断方法",
        "心肌梗死的急救措施",
        "心绞痛的特征",
        "心脏病的预防措施",
        "血压高的原因",
        "心脏病发作时怎么办？"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"查询: {query}")
        print('='*80)
        
        response = qa_system.process_query(query)
        
        print(f"回答: {response['answer']}")
        print(f"置信度: {response['confidence']:.3f}")
        print(f"引用数量: {len(response['citations'])}")
        print(f"查询分析: {response['query_analysis']['query_type']}")
        print(f"相关问题: {response['related_questions'][:3]}")

if __name__ == "__main__":
    main()