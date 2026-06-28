#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版医学文档解析器 - 支持精确溯源和宽泛问题回答
基于您的5个PDF教材构建完整的医学知识库
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import jieba
import jieba.posseg as pseg
from collections import Counter
import subprocess
import PyPDF2
import fitz  # PyMuPDF

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMedicalDocumentParser:
    def __init__(self, pdf_directory: str, output_directory: str):
        """初始化增强版医学文档解析器"""
        self.pdf_directory = Path(pdf_directory)
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        # 扩展的医学术语词典
        self.medical_terms = self._load_comprehensive_medical_terms()
        jieba.initialize()
        
        # 添加医学术语到jieba词典
        for term in self.medical_terms:
            jieba.add_word(term)
        
        # 常见医学同义词映射
        self.medical_synonyms = {
            '心脏病': ['心脏疾病', '心血管疾病', '心疾病'],
            '冠心病': ['冠状动脉粥样硬化', '冠状动脉疾病', '冠心病'],
            '心肌梗死': ['心肌梗死', '心梗', '急性心肌梗死'],
            '心力衰竭': ['心衰', '心功能不全', '心力衰竭'],
            '高血压': ['高血压病', '动脉高血压', '血压升高'],
            '心律失常': ['心律不齐', '心律紊乱', '心律失常'],
            '胸痛': ['胸痛', '心前区疼痛', '胸部疼痛'],
            '气短': ['呼吸困难', '气短', '气促'],
            '心悸': ['心跳加速', '心悸', '心慌']
        }
        
        # 问题扩展关键词
        self.query_expansion_keywords = {
            '症状': ['表现', '征象', '现象', '特征'],
            '诊断': ['检查', '确诊', '识别', '判断'],
            '治疗': ['处理', '治疗方案', '疗法', '药物'],
            '病因': ['原因', '致病因素', '诱因', '起源'],
            '预防': ['预防措施', '预防方法', '预防策略', '防护'],
            '预后': ['结局', '转归', '后果', '结果']
        }
    
    def _load_comprehensive_medical_terms(self) -> List[str]:
        """加载完整的医学术语词典"""
        return [
            # 基础医学术语
            "心脏", "心肌", "心房", "心室", "冠状动脉", "主动脉", "肺动脉",
            "血压", "心率", "心律", "心音", "心脏杂音", "心力衰竭", "心源性休克",
            
            # 疾病名称
            "冠心病", "心肌梗死", "心绞痛", "心力衰竭", "高血压", "心律失常",
            "先天性心脏病", "风湿性心脏病", "心肌炎", "心包炎", "感染性心内膜炎",
            "肺心病", "心脏瓣膜病", "主动脉瓣狭窄", "二尖瓣关闭不全",
            
            # 症状和体征
            "胸痛", "胸闷", "气短", "呼吸困难", "心悸", "水肿", "头晕", "晕厥",
            "乏力", "疲劳", "出汗", "恶心", "呕吐", "腹痛", "腰痛", "关节痛",
            
            # 检查方法
            "心电图", "超声心动图", "心脏彩超", "冠脉造影", "心导管检查", "运动负荷试验",
            "动态心电图", "心肌酶谱", "肌钙蛋白", "CK-MB", "血脂", "血糖",
            "胸部X线", "CT", "MRI", "核素扫描",
            
            # 药物名称
            "阿司匹林", "氯吡格雷", "替格瑞洛", "阿托伐他汀", "瑞舒伐他汀",
            "美托洛尔", "比索洛尔", "卡维地洛", "依那普利", "贝那普利",
            "缬沙坦", "氯沙坦", "氢氯噻嗪", "呋塞米", "螺内酯", "硝苯地平",
            "氨氯地平", "硝酸甘油", "单硝酸异山梨酯",
            
            # 数值和单位
            "mmHg", "mg/dL", "mmol/L", "bpm", "次/分", "mg", "g", "kg",
            
            # 医学概念
            "病因", "病理", "发病机制", "临床表现", "诊断标准", "治疗原则",
            "预后评估", "预防措施", "并发症", "鉴别诊断", "适应症", "禁忌症"
        ]
    
    def extract_text_with_coordinates_from_pdf(self, pdf_path: str) -> List[Dict]:
        """从PDF中提取文本并保留位置信息"""
        logger.info(f"解析PDF文件: {pdf_path}")
        
        pdf_document = fitz.open(pdf_path)
        pages_data = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text_dict = page.get_text("dict")
            
            page_data = {
                'page_number': page_num + 1,
                'text_blocks': []
            }
            
            for block in text_dict["blocks"]:
                if "lines" in block:  # 文本块
                    block_text = ""
                    sentences = []
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                block_text += text + " "
                                
                                # 检测句子边界
                                if text.endswith(('。', '！', '？', '.', '!', '?')):
                                    sentences.append({
                                        'text': text,
                                        'bbox': span["bbox"],
                                        'font_size': span["size"]
                                    })
                    
                    if block_text.strip():
                        page_data['text_blocks'].append({
                            'content': block_text.strip(),
                            'sentences': sentences,
                            'bbox': block["bbox"]
                        })
            
            pages_data.append(page_data)
        
        pdf_document.close()
        return pages_data
    
    def extract_medical_knowledge_comprehensive(self, pages_data: List[Dict], pdf_filename: str) -> List[Dict]:
        """全面提取医学知识"""
        all_chunks = []
        
        for page_data in pages_data:
            page_number = page_data['page_number']
            
            for block_idx, block in enumerate(page_data['text_blocks']):
                content = block['content']
                sentences = block['sentences']
                
                # 跳过过短的文本块
                if len(content) < 50:
                    continue
                
                # 检测医学知识类型
                knowledge_types = self._detect_knowledge_types(content)
                
                # 提取医学术语
                medical_terms_found = self._extract_medical_terms_enhanced(content)
                
                # 生成知识块
                chunk = {
                    'id': f"{pdf_filename}_page_{page_number}_block_{block_idx}",
                    'content': content,
                    'page_number': page_number,
                    'block_index': block_idx,
                    'source_file': pdf_filename,
                    'source_path': str(self.pdf_directory / pdf_filename),
                    'knowledge_types': knowledge_types,
                    'medical_terms': medical_terms_found,
                    'sentences': sentences,
                    'completeness_score': self._calculate_completeness(content, knowledge_types),
                    'difficulty_level': self._assess_content_difficulty(content),
                    'relevance_score': self._calculate_relevance_score(content, medical_terms_found)
                }
                
                all_chunks.append(chunk)
        
        return all_chunks
    
    def _detect_knowledge_types(self, content: str) -> List[str]:
        """检测医学知识类型"""
        knowledge_patterns = {
            '病因学': ['病因', '原因', '致病因素', '诱发因素', '发病原因'],
            '病理学': ['病理', '发病机制', '病理生理', '病理变化', '病理过程'],
            '症状学': ['症状', '临床表现', '体征', '表现', '症状和体征'],
            '诊断学': ['诊断', '检查', '诊断标准', '鉴别诊断', '确诊方法'],
            '治疗学': ['治疗', '治疗原则', '治疗方案', '药物治疗', '手术治疗'],
            '预防医学': ['预防', '预防措施', '预防策略', '预防方法'],
            '预后学': ['预后', '转归', '结局', '后果', '预后评估'],
            '药物学': ['药物', '剂量', '用法', '不良反应', '副作用', '禁忌症'],
            '流行病学': ['流行病学', '发病率', '患病率', '死亡率', '危险因素'],
            '解剖学': ['解剖', '结构', '解剖结构', '解剖关系', '解剖变异']
        }
        
        detected_types = []
        content_lower = content.lower()
        
        for knowledge_type, keywords in knowledge_patterns.items():
            if any(keyword in content for keyword in keywords):
                detected_types.append(knowledge_type)
        
        return detected_types
    
    def _extract_medical_terms_enhanced(self, content: str) -> List[str]:
        """增强的医学术语提取"""
        words = pseg.cut(content)
        found_terms = []
        
        for word, flag in words:
            # 检查是否是医学术语
            if word in self.medical_terms:
                found_terms.append(word)
            
            # 检查同义词
            for standard_term, synonyms in self.medical_synonyms.items():
                if word in synonyms:
                    found_terms.append(standard_term)
        
        return list(set(found_terms))
    
    def _calculate_completeness(self, content: str, knowledge_types: List[str]) -> float:
        """计算知识完整性分数"""
        expected_types = ['病因学', '病理学', '症状学', '诊断学', '治疗学']
        matched_types = len([t for t in expected_types if t in knowledge_types])
        return matched_types / len(expected_types)
    
    def _assess_content_difficulty(self, content: str) -> str:
        """评估内容难度"""
        advanced_indicators = ['复杂', '疑难', '罕见', '重型', '危重', '紧急']
        basic_indicators = ['基础', '简单', '常见', '一般', '典型']
        
        if any(indicator in content for indicator in advanced_indicators):
            return 'advanced'
        elif any(indicator in content for indicator in basic_indicators):
            return 'basic'
        else:
            return 'intermediate'
    
    def _calculate_relevance_score(self, content: str, medical_terms: List[str]) -> float:
        """计算内容相关性分数"""
        if not medical_terms:
            return 0.1
        
        # 基于医学术语密度计算
        term_density = len(medical_terms) / len(content.split()) * 10
        return min(term_density, 1.0)
    
    def expand_query_for_broad_questions(self, query: str) -> List[str]:
        """为宽泛问题生成扩展查询"""
        expanded_queries = [query]  # 原始查询
        
        # 检测查询中的关键概念
        key_concepts = self._extract_key_concepts(query)
        
        # 生成相关扩展查询
        for concept in key_concepts:
            # 添加相关医学概念
            if concept in self.query_expansion_keywords:
                for related in self.query_expansion_keywords[concept]:
                    expanded_queries.append(f"{concept} {related}")
                    expanded_queries.append(f"{related} {concept}")
            
            # 添加同义词扩展
            if concept in self.medical_synonyms:
                for synonym in self.medical_synonyms[concept]:
                    expanded_queries.append(synonym)
                    expanded_queries.append(f"{concept} {synonym}")
        
        # 生成更宽泛的查询
        if any(word in query for word in ['什么', '哪些', '怎么', '如何']):
            # 对于"什么"类型的问题，添加广泛的概念
            broad_terms = ['概念', '定义', '分类', '特点', '重要性']
            for term in broad_terms:
                expanded_queries.append(f"{query} {term}")
        
        return list(set(expanded_queries))  # 去重
    
    def _extract_key_concepts(self, query: str) -> List[str]:
        """提取查询中的关键概念"""
        words = jieba.cut(query)
        concepts = []
        
        for word in words:
            if word in self.medical_terms or word in [term for synonyms in self.medical_synonyms.values() for term in synonyms]:
                concepts.append(word)
        
        return concepts
    
    def build_comprehensive_knowledge_base(self) -> Dict:
        """构建全面的医学知识库"""
        logger.info("开始构建全面的医学知识库...")
        
        all_chunks = []
        processed_files = []
        
        # 处理所有PDF文件
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"处理文件: {pdf_file.name}")
                
                # 提取文本和位置信息
                pages_data = self.extract_text_with_coordinates_from_pdf(str(pdf_file))
                
                # 提取医学知识
                chunks = self.extract_medical_knowledge_comprehensive(pages_data, pdf_file.name)
                all_chunks.extend(chunks)
                
                processed_files.append({
                    'filename': pdf_file.name,
                    'pages': len(pages_data),
                    'chunks': len(chunks)
                })
                
                logger.info(f"从 {pdf_file.name} 提取了 {len(chunks)} 个知识块")
                
            except Exception as e:
                logger.error(f"处理文件 {pdf_file.name} 时出错: {e}")
                continue
        
        # 构建知识库元数据
        knowledge_base = {
            'metadata': {
                'created_at': '2025-12-26',
                'version': '2.0_enhanced',
                'total_chunks': len(all_chunks),
                'processed_files': processed_files,
                'medical_categories': list(set([term for chunk in all_chunks for term in chunk.get('medical_terms', [])])),
                'knowledge_types': list(set([kt for chunk in all_chunks for kt in chunk.get('knowledge_types', [])]))
            },
            'chunks': all_chunks,
            'statistics': {
                'total_pages': sum([f['pages'] for f in processed_files]),
                'average_chunks_per_page': len(all_chunks) / sum([f['pages'] for f in processed_files]) if processed_files else 0,
                'medical_terms_count': len(set([term for chunk in all_chunks for term in chunk.get('medical_terms', [])])),
                'knowledge_type_distribution': self._calculate_knowledge_type_distribution(all_chunks)
            }
        }
        
        # 保存知识库
        output_path = self.output_directory / "enhanced_medical_knowledge_base.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
        
        logger.info(f"增强版知识库已保存到: {output_path}")
        logger.info(f"总计处理了 {len(all_chunks)} 个知识块")
        
        return knowledge_base
    
    def _calculate_knowledge_type_distribution(self, chunks: List[Dict]) -> Dict:
        """计算知识类型分布"""
        type_counts = {}
        for chunk in chunks:
            for knowledge_type in chunk.get('knowledge_types', []):
                type_counts[knowledge_type] = type_counts.get(knowledge_type, 0) + 1
        return type_counts
    
    def generate_source_citation(self, chunk: Dict, sentence_text: str = None) -> Dict:
        """生成精确的源引用信息"""
        citation = {
            'source_file': chunk['source_file'],
            'page_number': chunk['page_number'],
            'block_index': chunk['block_index'],
            'source_path': chunk['source_path']
        }
        
        if sentence_text and chunk.get('sentences'):
            # 尝试找到对应的句子
            for sentence in chunk['sentences']:
                if sentence_text in sentence['text'] or sentence['text'] in sentence_text:
                    citation['sentence_bbox'] = sentence['bbox']
                    citation['font_size'] = sentence['font_size']
                    break
        
        return citation

def main():
    """主函数"""
    parser = EnhancedMedicalDocumentParser(
        pdf_directory="d:\\BS\\教材",
        output_directory="d:\\BS\\medical-qa-ragflow\\data"
    )
    
    # 构建增强版知识库
    knowledge_base = parser.build_comprehensive_knowledge_base()
    
    print(f"知识库构建完成！")
    print(f"总计知识块数: {knowledge_base['metadata']['total_chunks']}")
    print(f"处理文件数: {len(knowledge_base['metadata']['processed_files'])}")
    print(f"医学术语数: {knowledge_base['statistics']['medical_terms_count']}")

if __name__ == "__main__":
    main()