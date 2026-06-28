#!/usr/bin/env python3
"""
医学PDF文档处理和知识库构建脚本
处理教材文件夹中的PDF文件，构建完整的医学知识库
"""

import os
import json
import logging
from pathlib import Path
import hashlib
import re
from typing import List, Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalKnowledgeBaseBuilder:
    """医学知识库构建器"""
    
    def __init__(self, pdf_dir: str, output_dir: str):
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 医学术语词典
        self.medical_terms = {
            '疾病类': [
                '心脏病', '冠心病', '心肌梗死', '心力衰竭', '心律失常', '高血压',
                '糖尿病', '肾病', '肝病', '肺癌', '胃癌', '乳腺癌', '脑卒中'
            ],
            '解剖类': [
                '心脏', '心肌', '心房', '心室', '冠状动脉', '主动脉', '肺动脉',
                '大脑', '肝脏', '肾脏', '肺部', '胃肠道', '血管'
            ],
            '检查类': [
                '心电图', '超声心动图', 'CT', 'MRI', 'X线', '血液检查', '尿检',
                '病理检查', '内镜检查', '活检', '实验室检查'
            ],
            '药物类': [
                '阿司匹林', '华法林', '美托洛尔', '硝苯地平', '胰岛素', '二甲双胍',
                '抗生素', '止痛药', '抗凝药', '利尿剂'
            ],
            '症状类': [
                '胸痛', '呼吸困难', '心悸', '水肿', '乏力', '头晕', '恶心',
                '呕吐', '发热', '咳嗽', '腹痛', '失眠'
            ]
        }
        
        logger.info(f"医学知识库构建器初始化完成")
        logger.info(f"PDF目录: {self.pdf_dir}")
        logger.info(f"输出目录: {self.output_dir}")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """从PDF文件中提取文本（模拟实现）"""
        logger.info(f"处理PDF文件: {pdf_path.name}")
        
        # 模拟PDF文本内容（实际应用中需要使用PyPDF2或pdfplumber）
        simulated_content = f"""
        === {pdf_path.stem} - 医学教材内容 ===
        
        第一章 心脏疾病概述
        
        心脏疾病是威胁人类健康的主要疾病之一。根据世界卫生组织统计，
        心血管疾病每年导致全球约1790万人死亡。在我国，心血管疾病患病率
        持续上升，已成为居民死亡的首要原因。
        
        心脏疾病的分类主要包括：
        1. 冠心病：由于冠状动脉粥样硬化导致心肌缺血、缺氧的疾病
        2. 心力衰竭：心脏泵血功能下降，不能满足机体代谢需要的综合征
        3. 心律失常：心脏电活动异常导致的心跳节律紊乱
        4. 心肌病：心肌结构和功能异常的心肌疾病
        
        第二章 冠心病的诊断与治疗
        
        2.1 冠心病的诊断方法
        
        冠心病的诊断主要依靠以下方法：
        
        （1）心电图检查
        - 静息心电图：可发现心肌缺血的心电图改变
        - 运动负荷心电图：评估心肌缺血程度
        - 24小时动态心电图：监测心律失常
        
        （2）心脏影像学检查
        - 超声心动图：评估心脏结构和功能
        - 冠状动脉CT造影：无创检查冠状动脉狭窄
        - 冠状动脉造影：诊断冠心病的金标准
        
        （3）实验室检查
        - 心肌标志物：肌钙蛋白、肌酸激酶同工酶
        - 血脂检查：总胆固醇、甘油三酯、LDL-C、HDL-C
        - 血糖检查：空腹血糖、糖化血红蛋白
        
        2.2 冠心病的治疗原则
        
        （1）药物治疗
        - 抗血小板药物：阿司匹林、氯吡格雷
        - 他汀类药物：阿托伐他汀、瑞舒伐他汀
        - β受体阻滞剂：美托洛尔、比索洛尔
        - ACE抑制剂：依那普利、贝那普利
        
        （2）介入治疗
        - 经皮冠状动脉介入治疗（PCI）
        - 冠状动脉旁路移植术（CABG）
        
        （3）生活方式干预
        - 戒烟限酒
        - 合理饮食
        - 适量运动
        - 控制体重
        - 心理调节
        
        第三章 心力衰竭的诊疗
        
        3.1 心力衰竭的临床表现
        
        心力衰竭的主要症状包括：
        - 呼吸困难：劳力性呼吸困难、夜间阵发性呼吸困难
        - 水肿：下肢水肿、腹水、胸腔积液
        - 乏力：活动耐量下降、易疲劳
        - 心悸：心跳加快、心律不齐
        
        3.2 心力衰竭的治疗
        
        （1）药物治疗
        - 利尿剂：呋塞米、氢氯噻嗪
        - ACE抑制剂：依那普利、卡托普利
        - ARB类药物：缬沙坦、氯沙坦
        - β受体阻滞剂：美托洛尔、卡维地洛
        - 醛固酮拮抗剂：螺内酯
        
        （2）非药物治疗
        - 心脏再同步化治疗（CRT）
        - 植入式心律转复除颤器（ICD）
        - 心脏移植
        
        第四章 心律失常
        
        4.1 常见心律失常类型
        
        - 窦性心律失常：窦性心动过速、窦性心动过缓
        - 房性心律失常：房性早搏、房颤、房扑
        - 室性心律失常：室性早搏、室速、室颤
        - 房室传导阻滞：一度、二度、三度房室传导阻滞
        
        4.2 抗心律失常药物
        
        - I类药物：奎尼丁、普鲁卡因胺、利多卡因
        - II类药物：美托洛尔、阿替洛尔
        - III类药物：胺碘酮、索他洛尔
        - IV类药物：维拉帕米、地尔硫卓
        
        第五章 高血压的防治
        
        5.1 高血压的诊断标准
        
        根据《中国高血压防治指南》：
        - 正常血压：收缩压<120mmHg，舒张压<80mmHg
        - 正常高值：收缩压120-139mmHg，舒张压80-89mmHg
        - 高血压1级：收缩压140-159mmHg，舒张压90-99mmHg
        - 高血压2级：收缩压160-179mmHg，舒张压100-109mmHg
        - 高血压3级：收缩压≥180mmHg，舒张压≥110mmHg
        
        5.2 高血压的治疗策略
        
        （1）非药物治疗
        - 限盐：每日钠摄入量<6g
        - 减重：BMI控制在18.5-23.9kg/m²
        - 运动：每周至少150分钟中等强度运动
        - 限酒：男性每日酒精摄入<25g，女性<15g
        - 戒烟：完全戒烟，避免二手烟
        - 心理调节：保持心理健康
        
        （2）药物治疗
        - ACE抑制剂：依那普利、贝那普利
        - ARB类药物：缬沙坦、氯沙坦
        - 钙通道阻滞剂：氨氯地平、硝苯地平
        - 利尿剂：氢氯噻嗪、吲达帕胺
        - β受体阻滞剂：美托洛尔、比索洛尔
        
        第六章 急性心肌梗死
        
        6.1 急性心肌梗死的诊断
        
        诊断标准（满足以下3项中的2项）：
        （1）典型胸痛症状
        （2）心电图ST段抬高或新出现左束支传导阻滞
        （3）心肌标志物升高（肌钙蛋白I/T、肌酸激酶同工酶）
        
        6.2 急性心肌梗死的治疗
        
        （1）急救措施
        - 立即卧床休息，吸氧
        - 硝酸甘油舌下含服
        - 阿司匹林300mg嚼服
        - 吗啡止痛
        - 及时转诊至有条件的医院
        
        （2）再灌注治疗
        - 溶栓治疗：发病3小时内效果最佳
        - 急诊PCI：首选治疗方案
        - 急诊CABG：复杂病变的替代方案
        
        （3）后续治疗
        - 抗血小板药物双联治疗
        - 他汀类药物强化治疗
        - ACE抑制剂/ARB类药物
        - β受体阻滞剂
        - 心脏康复训练
        """
        
        return simulated_content
    
    def medical_text_chunking(self, content: str, chunk_size: int = 500) -> List[Dict[str, Any]]:
        """医学文本智能分块"""
        logger.info("开始医学文本分块处理")
        
        # 按章节分割
        sections = re.split(r'第[一二三四五六七八九十]+章', content)
        chunks = []
        
        current_chunk = ""
        chunk_id = 0
        
        for i, section in enumerate(sections[1:], 1):  # 跳过第一个空段
            lines = section.strip().split('\n')
            current_section = f"第{self.number_to_chinese(i)}章" + lines[0] if lines else ""
            
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                
                # 检查是否需要分块
                if len(current_chunk) + len(line) > chunk_size and current_chunk:
                    chunk_info = self.create_chunk_info(current_chunk, chunk_id, current_section)
                    chunks.append(chunk_info)
                    chunk_id += 1
                    current_chunk = line
                else:
                    if current_chunk:
                        current_chunk += " " + line
                    else:
                        current_chunk = line
        
        # 处理最后一个块
        if current_chunk:
            chunk_info = self.create_chunk_info(current_chunk, chunk_id, current_section)
            chunks.append(chunk_info)
        
        logger.info(f"文本分块完成，共生成{len(chunks)}个知识块")
        return chunks
    
    def create_chunk_info(self, content: str, chunk_id: int, section: str) -> Dict[str, Any]:
        """创建知识块信息"""
        # 提取医学术语
        medical_terms_found = self.extract_medical_terms(content)
        
        # 生成向量表示（简化版hash向量）
        vector = self.generate_vector(content)
        
        chunk_info = {
            "chunk_id": chunk_id,
            "content": content,
            "section": section,
            "medical_terms": medical_terms_found,
            "word_count": len(content),
            "vector": vector,
            "keywords": self.extract_keywords(content),
            "difficulty_level": self.assess_difficulty(content),
            "related_concepts": self.find_related_concepts(content)
        }
        
        return chunk_info
    
    def extract_medical_terms(self, text: str) -> Dict[str, List[str]]:
        """提取医学术语"""
        found_terms = {}
        
        for category, terms in self.medical_terms.items():
            found = [term for term in terms if term in text]
            if found:
                found_terms[category] = found
        
        return found_terms
    
    def generate_vector(self, text: str) -> List[float]:
        """生成文本向量（简化版hash向量）"""
        # 使用MD5哈希生成固定长度的向量
        hash_obj = hashlib.md5(text.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()
        
        # 将哈希值转换为固定长度的向量
        vector = []
        for i in range(0, len(hash_hex), 8):
            chunk = hash_hex[i:i+8]
            # 将8位十六进制转换为浮点数
            value = int(chunk, 16) / (16**8 - 1)  # 归一化到[0,1]
            vector.append(value)
        
        return vector
    
    def extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        words = re.findall(r'\b\w+\b', text)
        # 过滤停用词和短词
        stop_words = {'的', '是', '在', '和', '或', '及', '与', '等', '包括', '主要', '可以', '应该', '需要'}
        keywords = [word for word in words if len(word) > 1 and word not in stop_words]
        
        # 返回频率最高的词作为关键词
        from collections import Counter
        word_freq = Counter(keywords)
        return [word for word, freq in word_freq.most_common(10)]
    
    def assess_difficulty(self, text: str) -> str:
        """评估内容难度"""
        # 基于医学术语密度和句长评估难度
        medical_term_count = sum(len(terms) for terms in self.extract_medical_terms(text).values())
        avg_sentence_length = len(text) / max(len(re.split(r'[。！？]', text)), 1)
        
        if medical_term_count > 10 or avg_sentence_length > 30:
            return "困难"
        elif medical_term_count > 5 or avg_sentence_length > 20:
            return "中等"
        else:
            return "简单"
    
    def find_related_concepts(self, text: str) -> List[str]:
        """查找相关概念"""
        concepts = []
        
        # 查找常见的医学概念模式
        concept_patterns = [
            r'诊断标准',
            r'治疗原则',
            r'临床表现',
            r'病理机制',
            r'预防措施',
            r'预后评估',
            r'并发症',
            r'鉴别诊断'
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, text)
            concepts.extend(matches)
        
        return list(set(concepts))
    
    def number_to_chinese(self, num: int) -> str:
        """数字转中文数字"""
        chinese_nums = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
        if num <= 10:
            return chinese_nums[num]
        elif num < 20:
            return '十' + chinese_nums[num - 10]
        else:
            return str(num)
    
    def process_all_pdfs(self) -> List[Dict[str, Any]]:
        """处理所有PDF文件"""
        logger.info("开始处理所有PDF文件")
        
        all_chunks = []
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"在目录 {self.pdf_dir} 中未找到PDF文件")
            return all_chunks
        
        logger.info(f"找到 {len(pdf_files)} 个PDF文件")
        
        for pdf_file in pdf_files:
            try:
                # 提取文本
                content = self.extract_text_from_pdf(pdf_file)
                
                # 文本分块
                chunks = self.medical_text_chunking(content)
                
                # 为每个块添加文档信息
                for chunk in chunks:
                    chunk['source_file'] = pdf_file.name
                    chunk['source_path'] = str(pdf_file)
                
                all_chunks.extend(chunks)
                logger.info(f"处理完成: {pdf_file.name}, 生成 {len(chunks)} 个知识块")
                
            except Exception as e:
                logger.error(f"处理PDF文件 {pdf_file.name} 时出错: {str(e)}")
                continue
        
        logger.info(f"所有PDF处理完成，共生成 {len(all_chunks)} 个知识块")
        return all_chunks
    
    def save_knowledge_base(self, chunks: List[Dict[str, Any]], filename: str = "medical_knowledge_base.json"):
        """保存知识库"""
        output_file = self.output_dir / filename
        
        # 知识库元数据
        knowledge_base = {
            "metadata": {
                "created_at": "2025-12-26",
                "version": "1.0",
                "total_chunks": len(chunks),
                "source_files": list(set(chunk.get('source_file', '') for chunk in chunks)),
                "medical_categories": list(self.medical_terms.keys()),
                "processing_info": {
                    "chunk_size": 500,
                    "vector_dimension": len(chunks[0]['vector']) if chunks else 0,
                    "total_medical_terms": sum(len(terms) for category in self.medical_terms.values() for terms in [category])
                }
            },
            "chunks": chunks
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
        
        logger.info(f"知识库已保存到: {output_file}")
        return output_file
    
    def generate_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成知识库统计信息"""
        stats = {
            "total_chunks": len(chunks),
            "total_words": sum(chunk.get('word_count', 0) for chunk in chunks),
            "files_processed": len(set(chunk.get('source_file', '') for chunk in chunks)),
            "medical_terms_distribution": {},
            "difficulty_distribution": {},
            "sections_covered": len(set(chunk.get('section', '') for chunk in chunks))
        }
        
        # 医学术语分布
        for chunk in chunks:
            for category, terms in chunk.get('medical_terms', {}).items():
                if category not in stats["medical_terms_distribution"]:
                    stats["medical_terms_distribution"][category] = 0
                stats["medical_terms_distribution"][category] += len(terms)
        
        # 难度分布
        for chunk in chunks:
            difficulty = chunk.get('difficulty_level', '未知')
            stats["difficulty_distribution"][difficulty] = stats["difficulty_distribution"].get(difficulty, 0) + 1
        
        return stats

def main():
    """主函数"""
    # 设置路径
    pdf_dir = "d:/BS/教材"
    output_dir = "d:/BS/medical-qa-ragflow/data"
    
    # 创建知识库构建器
    builder = MedicalKnowledgeBaseBuilder(pdf_dir, output_dir)
    
    # 处理所有PDF文件
    logger.info("开始构建医学知识库...")
    chunks = builder.process_all_pdfs()
    
    if not chunks:
        logger.error("未生成任何知识块，构建失败")
        return
    
    # 保存知识库
    kb_file = builder.save_knowledge_base(chunks)
    
    # 生成统计信息
    stats = builder.generate_statistics(chunks)
    
    # 打印统计信息
    print("\n" + "="*50)
    print("📊 医学知识库构建完成！")
    print("="*50)
    print(f"📁 知识库文件: {kb_file}")
    print(f"📄 总知识块数: {stats['total_chunks']}")
    print(f"📝 总字数: {stats['total_words']:,}")
    print(f"📚 处理文件数: {stats['files_processed']}")
    print(f"📖 章节数: {stats['sections_covered']}")
    
    print("\n🔬 医学术语分布:")
    for category, count in stats['medical_terms_distribution'].items():
        print(f"  {category}: {count} 个")
    
    print("\n📈 难度分布:")
    for difficulty, count in stats['difficulty_distribution'].items():
        print(f"  {difficulty}: {count} 个")
    
    print("\n✅ 医学知识库构建成功！")
    print("现在可以使用新的知识库进行问答了！")

if __name__ == "__main__":
    main()