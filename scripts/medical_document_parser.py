#!/usr/bin/env python3
"""
医学教材深度解析器
基于RAGFlow的医学PDF智能解析与分块系统
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import jieba
import jieba.posseg as pseg

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalDocumentParser:
    def __init__(self, config_path: str):
        """初始化医学文档解析器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 加载医学术语词典
        self.medical_terms = self._load_medical_terms()
        jieba.initialize()
        
        # 添加医学术语到jieba词典
        for term in self.medical_terms:
            jieba.add_word(term)
    
    def _load_medical_terms(self) -> List[str]:
        """加载医学术语词典"""
        medical_terms = [
            # 疾病相关
            "病因", "病理", "发病机制", "临床表现", "症状", "体征", "诊断", "鉴别诊断",
            "治疗", "预后", "预防", "并发症", "转归", "康复", "护理", "监护",
            
            # 解剖结构
            "心脏", "肺脏", "肝脏", "肾脏", "胃肠道", "神经系统", "内分泌系统",
            "免疫系统", "血液系统", "骨骼系统", "肌肉系统", "皮肤", "眼睛", "耳朵",
            
            # 检查方法
            "实验室检查", "影像学检查", "超声检查", "CT", "MRI", "X线", "心电图",
            "脑电图", "内镜检查", "活检", "病理检查", "免疫检查", "基因检测",
            
            # 药物相关
            "抗生素", "抗炎药", "止痛药", "降压药", "降糖药", "抗凝药", "利尿剂",
            "血管紧张素", "β受体阻滞剂", "钙通道阻滞剂", "剂量", "用法", "不良反应",
            
            # 数值和单位
            "mmHg", "mg/dL", "mmol/L", "μmol/L", "IU/L", "g/L", "pg/mL", "ng/mL",
            "ml/min", "bpm", "次/分", "次/日", "mg", "g", "kg", "cm", "mm", "μm"
        ]
        return medical_terms
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从PDF中提取文本（模拟实现）"""
        logger.info(f"解析PDF文件: {pdf_path}")
        
        # 这里应该是实际的PDF解析逻辑
        # 由于环境限制，我们模拟解析过程
        sample_text = """
        第1章 心脏疾病概述
        
        1.1 病因与病理
        
        心脏疾病的病因多种多样，主要包括先天性因素和后天性因素。先天性心脏病是由于胚胎期心脏发育异常所致，而后天性心脏病则多由感染、缺血、创伤等因素引起。
        
        病理变化方面，不同类型的心脏病有不同的病理特征。例如，冠心病主要表现为冠状动脉粥样硬化，导致心肌缺血缺氧。病理检查可见血管内膜增厚、脂质沉积、平滑肌细胞增生等改变。
        
        1.2 临床表现
        
        心脏疾病的临床表现主要包括以下几个方面：
        
        1. 症状：
           - 胸痛：这是冠心病最常见的症状，多在活动后加重
           - 气短：心功能不全时出现，初期仅在剧烈活动时出现
           - 心悸：心律失常的常见表现
           - 水肿：右心功能不全时出现下肢水肿
        
        2. 体征：
           - 心率改变：心动过速或心动过缓
           - 心律不齐：各种心律失常的体征
           - 心脏杂音：瓣膜病变的特征性表现
           - 颈静脉怒张：右心功能不全的体征
        
        1.3 诊断
        
        心脏疾病的诊断需要结合临床表现、实验室检查和影像学检查：
        
        1. 实验室检查：
           - 心肌酶谱：CK、CK-MB、cTnI、cTnT
           - 血脂：TC、TG、LDL-C、HDL-C
           - 血糖：空腹血糖、糖化血红蛋白
        
        2. 影像学检查：
           - 心电图：可显示心律失常、心肌缺血等改变
           - 超声心动图：评估心脏结构和功能
           - 冠脉造影：诊断冠心病的金标准
        
        1.4 治疗原则
        
        心脏疾病的治疗应根据具体病因和病情制定个体化方案：
        
        1. 药物治疗：
           - 抗血小板药物：阿司匹林、氯吡格雷
           - 他汀类药物：阿托伐他汀、瑞舒伐他汀
           - β受体阻滞剂：美托洛尔、比索洛尔
           - ACEI/ARB：依那普利、缬沙坦
        
        2. 介入治疗：
           - 经皮冠状动脉介入治疗（PCI）
           - 射频消融术
           - 心脏起搏器植入
        
        3. 外科治疗：
           - 冠状动脉旁路移植术（CABG）
           - 心脏瓣膜置换术
           - 心脏移植
        
        1.5 预后与预防
        
        心脏疾病的预后取决于多种因素，包括病因、病情严重程度、治疗及时性等。早期诊断和及时治疗是改善预后的关键。
        
        预防措施包括：
        - 控制危险因素：高血压、糖尿病、高脂血症
        - 健康生活方式：戒烟、限酒、合理饮食、适量运动
        - 定期体检：早期发现和治疗心血管疾病
        
        表1-1 主要心血管疾病及治疗药物
        
        | 疾病类型 | 主要症状 | 首选药物 | 剂量 |
        |---------|---------|---------|------|
        | 冠心病 | 胸痛、气短 | 硝酸甘油 | 0.5mg舌下含服 |
        | 心力衰竭 | 水肿、气短 | 利尿剂 | 呋塞米20-40mg/d |
        | 高血压 | 头痛、头晕 | ACEI | 依那普利5-20mg/d |
        """
        
        return sample_text
    
    def detect_chapters_and_sections(self, text: str) -> List[Dict]:
        """检测章节结构"""
        lines = text.split('\n')
        structure = []
        current_chapter = None
        current_section = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 检测章节标题
            chapter_match = re.match(r'^第(\d+)章\s+(.+)', line)
            if chapter_match:
                chapter_num = chapter_match.group(1)
                chapter_title = chapter_match.group(2)
                current_chapter = {
                    'type': 'chapter',
                    'number': int(chapter_num),
                    'title': chapter_title,
                    'start_line': i,
                    'content': []
                }
                structure.append(current_chapter)
                continue
            
            # 检测小节标题
            section_match = re.match(r'^(\d+\.\d*)\s+(.+)', line)
            if section_match:
                section_num = section_match.group(1)
                section_title = section_match.group(2)
                current_section = {
                    'type': 'section',
                    'number': section_num,
                    'title': section_title,
                    'start_line': i,
                    'content': []
                }
                if current_chapter:
                    current_chapter['content'].append(current_section)
                continue
            
            # 检测医学知识模式
            medical_pattern = self._detect_medical_knowledge_pattern(line)
            if medical_pattern:
                if current_section:
                    current_section['content'].append({
                        'type': 'medical_knowledge',
                        'pattern': medical_pattern,
                        'text': line,
                        'line_number': i
                    })
        
        return structure
    
    def _detect_medical_knowledge_pattern(self, text: str) -> Optional[str]:
        """检测医学知识模式"""
        medical_patterns = {
            '病因': ['病因', '原因', '致病因素'],
            '病理': ['病理', '发病机制', '病理生理', '病理变化'],
            '临床表现': ['临床表现', '症状', '体征', '症状和体征'],
            '诊断': ['诊断', '检查', '鉴别诊断'],
            '治疗': ['治疗', '药物治疗', '手术治疗', '治疗方案'],
            '预后': ['预后', '转归', ' prognosis'],
            '预防': ['预防', '预防措施', '预防策略'],
            '表格数据': ['表', '表格', 'Table'],
            '药物信息': ['剂量', '用法', '不良反应', '禁忌症']
        }
        
        for pattern_name, keywords in medical_patterns.items():
            if any(keyword in text for keyword in keywords):
                return pattern_name
        
        return None
    
    def semantic_chunking(self, text: str, structure: List[Dict]) -> List[Dict]:
        """基于医学知识完整性的语义分块"""
        chunks = []
        
        for chapter in structure:
            if chapter['type'] == 'chapter':
                chapter_content = []
                for item in chapter['content']:
                    if item['type'] == 'section':
                        section_content = item['content']
                        
                        # 检测知识链完整性
                        knowledge_chains = self._identify_knowledge_chains(section_content)
                        
                        for chain in knowledge_chains:
                            chunk = {
                                'id': f"chunk_{len(chunks)}",
                                'type': 'medical_knowledge_chain',
                                'chapter': chapter['title'],
                                'section': item['title'],
                                'knowledge_patterns': [k['pattern'] for k in chain],
                                'content': ' '.join([k['text'] for k in chain]),
                                'completeness_score': self._calculate_completeness(chain),
                                'metadata': {
                                    'source_type': 'medical_textbook',
                                    'page_number': 'N/A',  # PDF解析时提供
                                    'document_title': '心脏疾病学',
                                    'line_numbers': [k['line_number'] for k in chain]
                                }
                            }
                            chunks.append(chunk)
        
        return chunks
    
    def _identify_knowledge_chains(self, section_content: List[Dict]) -> List[List[Dict]]:
        """识别医学知识链"""
        # 定义标准医学知识链模式
        standard_chains = [
            ['病因', '病理', '临床表现'],
            ['病因', '病理', '临床表现', '诊断'],
            ['临床表现', '诊断', '治疗'],
            ['诊断', '治疗', '预后'],
            ['病因', '预防']
        ]
        
        knowledge_chains = []
        current_chain = []
        
        for item in section_content:
            pattern = item['pattern']
            
            if pattern in ['病因', '病理', '临床表现', '诊断', '治疗', '预后', '预防']:
                current_chain.append(item)
                
                # 检查是否形成完整知识链
                if self._is_complete_chain(current_chain, standard_chains):
                    knowledge_chains.append(current_chain.copy())
                    current_chain = []
            else:
                # 非标准医学知识模式，单独作为块
                if current_chain:
                    knowledge_chains.append(current_chain.copy())
                    current_chain = []
                knowledge_chains.append([item])
        
        # 处理剩余的链
        if current_chain:
            knowledge_chains.append(current_chain)
        
        return knowledge_chains
    
    def _is_complete_chain(self, chain: List[Dict], standard_chains: List[List[str]]) -> bool:
        """检查是否为完整的知识链"""
        chain_patterns = [item['pattern'] for item in chain]
        
        for standard_chain in standard_chains:
            if len(chain_patterns) >= len(standard_chain):
                # 检查是否有足够匹配的模式
                matched_patterns = 0
                for pattern in chain_patterns:
                    if pattern in standard_chain:
                        matched_patterns += 1
                
                # 如果匹配率达到80%以上，认为是完整链
                if matched_patterns / len(standard_chain) >= 0.8:
                    return True
        
        return False
    
    def _calculate_completeness(self, chain: List[Dict]) -> float:
        """计算知识链完整性分数"""
        expected_patterns = ['病因', '病理', '临床表现', '诊断', '治疗', '预后']
        actual_patterns = [item['pattern'] for item in chain]
        
        matched = sum(1 for pattern in expected_patterns if pattern in actual_patterns)
        return matched / len(expected_patterns)
    
    def enhance_chunks_with_metadata(self, chunks: List[Dict]) -> List[Dict]:
        """为分块添加元数据增强"""
        for chunk in chunks:
            # 添加医学术语标注
            chunk['medical_terms'] = self._extract_medical_terms(chunk['content'])
            
            # 添加关键词提取
            chunk['keywords'] = self._extract_keywords(chunk['content'])
            
            # 添加难度级别评估
            chunk['difficulty_level'] = self._assess_difficulty(chunk['content'])
            
            # 添加相关性标签
            chunk['relevance_tags'] = self._generate_relevance_tags(chunk)
        
        return chunks
    
    def _extract_medical_terms(self, text: str) -> List[str]:
        """提取医学术语"""
        words = pseg.cut(text)
        medical_words = []
        
        for word, flag in words:
            if word in self.medical_terms or len(word) > 1:
                medical_words.append(word)
        
        return list(set(medical_words))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        words = jieba.cut(text)
        keywords = []
        
        for word in words:
            if len(word) > 1 and word not in ['的', '是', '在', '和', '或', '等']:
                keywords.append(word)
        
        # 返回频率最高的5个关键词
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(5)]
    
    def _assess_difficulty(self, text: str) -> str:
        """评估内容难度"""
        # 简单的难度评估逻辑
        if '基础' in text or '概述' in text:
            return 'basic'
        elif '复杂' in text or '疑难' in text:
            return 'advanced'
        else:
            return 'intermediate'
    
    def _generate_relevance_tags(self, chunk: Dict) -> List[str]:
        """生成相关性标签"""
        tags = []
        
        content = chunk['content']
        patterns = chunk.get('knowledge_patterns', [])
        
        # 基于知识模式生成标签
        for pattern in patterns:
            tags.append(f"knowledge_{pattern}")
        
        # 基于内容特征生成标签
        if '药物' in content or '治疗' in content:
            tags.append('treatment')
        if '诊断' in content or '检查' in content:
            tags.append('diagnosis')
        if '症状' in content or '表现' in content:
            tags.append('symptoms')
        
        return list(set(tags))
    
    def process_document(self, pdf_path: str) -> List[Dict]:
        """完整的文档处理流程"""
        logger.info(f"开始处理医学文档: {pdf_path}")
        
        # 1. 提取文本
        text = self.extract_text_from_pdf(pdf_path)
        
        # 2. 检测章节结构
        structure = self.detect_chapters_and_sections(text)
        
        # 3. 语义分块
        chunks = self.semantic_chunking(text, structure)
        
        # 4. 增强元数据
        enhanced_chunks = self.enhance_chunks_with_metadata(chunks)
        
        logger.info(f"文档处理完成，生成 {len(enhanced_chunks)} 个知识块")
        
        return enhanced_chunks

def main():
    """主函数"""
    parser = MedicalDocumentParser('./configs/medical_chunking_config.json')
    
    # 处理示例PDF
    pdf_path = './data/pdfs/sample_medical_textbook.pdf'
    chunks = parser.process_document(pdf_path)
    
    # 保存结果
    output_path = './data/processed_chunks.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    logger.info(f"处理结果已保存到: {output_path}")
    print(f"生成了 {len(chunks)} 个知识块")

if __name__ == "__main__":
    main()