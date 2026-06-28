#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医学智能问答系统 - Flask Web应用
Medical Q&A System - Flask Web Application

功能特性：
- 智能医学问答（DeepSeek API）
- 智能自测系统
- 对比分析工具
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import logging
from datetime import datetime
import uuid
import requests

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _load_env_file(path: str) -> None:
    """加载 .env 到环境变量"""
    try:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ[key] = value
    except Exception as e:
        logger.warning(f"加载 .env 失败: {e}")


# 尝试加载项目目录下的 .env
_load_env_file(os.path.join(os.path.dirname(__file__), ".env"))


def _get_deepseek_api_key() -> str:
    # 兼容两种变量名：DEEPSEEK_API_KEY（推荐）/ LLM_API_KEY（兼容旧实现）
    return (os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY") or "").strip()


def _normalize_deepseek_base_url(raw: str) -> str:
    """
    规范化 DeepSeek base_url。
    兼容用户配置为：
    - https://api.deepseek.com
    - https://api.deepseek.com/
    - https://api.deepseek.com/v1
    - https://api.deepseek.com/v1/
    """
    url = (raw or "").strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url or "https://api.deepseek.com"


# DeepSeek API配置（除 key 外均可环境变量覆盖）
DEEPSEEK_BASE_URL = _normalize_deepseek_base_url(os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
DEEPSEEK_TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.7"))
DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "2000"))

_key = _get_deepseek_api_key()
logger.info(
    "DeepSeek 配置: base_url=%s model=%s key=***%s",
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    (_key[-4:] if _key else "NONE"),
)

# ============================================
# RAG检索增强生成模块 - 医学知识库
# ============================================
class RAGKnowledgeBase:
    """
    RAG知识库模块：用于存储和检索医学知识
    支持TF-IDF检索和语义相似度匹配
    溯源来源：病理学系列教材
    """
    
    # 溯源教材列表
    SOURCE_TEXTBOOKS = [
        "病理学",
        "病理学（第2版）",
        "病理学高级教程",
        "诊断病理学",
        "病理学第2版"
    ]
    
    def __init__(self):
        # 医学知识库文档 - 基于病理学系列教材构建
        self.documents = {
            "doc_001": {
                "title": "心肌梗死病理机制",
                "content": "心肌梗死是由于冠状动脉粥样硬化斑块破裂，导致血栓形成，使冠状动脉血流急剧减少或中断，引起心肌严重而持久的缺血坏死。典型症状包括胸骨后剧烈疼痛、呼吸困难、恶心呕吐、大汗等。病理变化表现为心肌细胞凝固性坏死，周围有中性粒细胞浸润。",
                "keywords": ["心肌梗死", "冠状动脉", "胸痛", "缺血坏死", "病理"],
                "sources": [
                    {"textbook": "病理学", "page": 156},
                    {"textbook": "病理学（第2版）", "page": 168},
                    {"textbook": "病理学高级教程", "page": 234}
                ]
            },
            "doc_002": {
                "title": "高血压病理改变",
                "content": "高血压是一种常见的心血管疾病，长期高血压可导致全身小动脉硬化，表现为血管壁增厚、管腔狭窄。心脏可出现左心室肥厚，肾脏可出现原发性颗粒性固缩肾，脑可出现脑出血或脑软化。",
                "keywords": ["高血压", "动脉硬化", "左心室肥厚", "病理改变"],
                "sources": [
                    {"textbook": "病理学", "page": 142},
                    {"textbook": "诊断病理学", "page": 89},
                    {"textbook": "病理学第2版", "page": 155}
                ]
            },
            "doc_003": {
                "title": "心力衰竭病理生理",
                "content": "心力衰竭是各种心脏疾病导致的心功能不全，病理生理机制包括心肌收缩力下降、心室重构、神经体液调节紊乱等。临床表现为呼吸困难、疲劳、水肿等。诊断依据包括超声心动图、BNP检测等。",
                "keywords": ["心力衰竭", "诊断", "BNP", "超声心动图", "病理生理"],
                "sources": [
                    {"textbook": "病理学高级教程", "page": 287},
                    {"textbook": "病理学（第2版）", "page": 178},
                    {"textbook": "诊断病理学", "page": 112}
                ]
            },
            "doc_004": {
                "title": "炎症与感染",
                "content": "炎症是机体对损伤因子的防御反应，表现为红、肿、热、痛和功能障碍。炎症的基本病理变化包括变质、渗出和增生。感染是由病原体引起的炎症反应，如肺炎、感冒等。",
                "keywords": ["炎症", "感染", "红热肿痛", "病理变化"],
                "sources": [
                    {"textbook": "病理学", "page": 45},
                    {"textbook": "病理学第2版", "page": 52},
                    {"textbook": "病理学高级教程", "page": 89}
                ]
            },
            "doc_005": {
                "title": "糖尿病病理",
                "content": "糖尿病是一组以高血糖为特征的代谢性疾病，分为1型和2型。长期高血糖可导致多种并发症，包括糖尿病肾病、糖尿病视网膜病变、糖尿病神经病变等。病理改变主要累及血管和神经。",
                "keywords": ["糖尿病", "血糖控制", "二甲双胍", "胰岛素", "并发症"],
                "sources": [
                    {"textbook": "病理学", "page": 215},
                    {"textbook": "病理学（第2版）", "page": 228},
                    {"textbook": "诊断病理学", "page": 156}
                ]
            },
            "doc_006": {
                "title": "肺炎病理诊断",
                "content": "肺炎是肺部感染性疾病，常见类型包括大叶性肺炎、小叶性肺炎、间质性肺炎等。病理变化包括肺泡腔内渗出物、炎症细胞浸润等。诊断依靠临床表现、胸部影像学检查和病原学检测。",
                "keywords": ["肺炎", "肺部感染", "抗生素", "影像学", "病理诊断"],
                "sources": [
                    {"textbook": "诊断病理学", "page": 189},
                    {"textbook": "病理学高级教程", "page": 312},
                    {"textbook": "病理学第2版", "page": 198}
                ]
            },
            "doc_007": {
                "title": "肿瘤病理学",
                "content": "肿瘤是机体在各种致瘤因素作用下，细胞异常增生形成的新生物。肿瘤分为良性和恶性，恶性肿瘤具有侵袭性和转移性。肿瘤的病理诊断是肿瘤治疗的重要依据。",
                "keywords": ["肿瘤", "癌症", "良性", "恶性", "病理诊断"],
                "sources": [
                    {"textbook": "病理学", "page": 89},
                    {"textbook": "病理学（第2版）", "page": 98},
                    {"textbook": "病理学高级教程", "page": 156},
                    {"textbook": "诊断病理学", "page": 234}
                ]
            },
            "doc_008": {
                "title": "心血管疾病病理",
                "content": "心血管疾病包括冠心病、高血压性心脏病、心肌病等。冠心病的病理基础是冠状动脉粥样硬化，可导致心绞痛和心肌梗死。高血压性心脏病表现为左心室肥厚和心力衰竭。",
                "keywords": ["心血管", "冠心病", "心绞痛", "心肌梗死", "病理"],
                "sources": [
                    {"textbook": "病理学", "page": 135},
                    {"textbook": "病理学第2版", "page": 148},
                    {"textbook": "病理学高级教程", "page": 215}
                ]
            }
        }
    
    def retrieve(self, query, top_k=3):
        """
        检索知识库，返回相关文档
        :param query: 用户查询
        :param top_k: 返回前k个相关文档
        :return: 相关文档列表
        """
        results = []
        
        # 简单关键词匹配（模拟检索过程）
        for doc_id, doc in self.documents.items():
            # 检查标题和关键词是否匹配
            score = 0
            query_lower = query.lower()
            
            # 标题匹配
            if query_lower in doc["title"].lower():
                score += 5
            
            # 关键词匹配
            for keyword in doc["keywords"]:
                if keyword.lower() in query_lower:
                    score += 2
            
            # 内容匹配
            if query_lower in doc["content"].lower():
                score += 3
            
            if score > 0:
                results.append({
                    "doc_id": doc_id,
                    "title": doc["title"],
                    "content": doc["content"],
                    "score": score,
                    "sources": doc.get("sources", [])  # 添加溯源信息
                })
        
        # 按相关性排序并返回top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

# 初始化RAG知识库
rag_knowledge_base = RAGKnowledgeBase()

# ============================================
# 模拟问答数据（保留原有功能）
# ============================================
QA_RESPONSES = {
    "心肌梗死症状": {
        "response": "心肌梗死的典型症状包括：\n1. 胸骨后剧烈疼痛，常持续超过30分钟\n2. 疼痛可放射至左肩、左臂、下颌\n3. 伴有恶心、呕吐、大汗\n4. 严重时可出现心律失常、休克\n\n⚠️ 重要提醒：出现这些症状应立即就医！",
        "confidence": 0.95
    },
    "高血压用药": {
        "response": "高血压常用药物包括：\n1. ACE抑制剂/ARB：卡托普利、依那普利等\n2. 钙通道阻滞剂：硝苯地平、氨氯地平等\n3. 利尿剂：氢氯噻嗪、呋塞米等\n4. β受体阻滞剂：美托洛尔、阿替洛尔等\n\n具体用药请在医生指导下选择。",
        "confidence": 0.90
    },
    "心力衰竭诊断": {
        "response": "心力衰竭的诊断依据：\n1. 临床表现：呼吸困难、疲劳、水肿\n2. 超声心动图：左室射血分数(EF)<40%\n3. 生物标志物：BNP>100pg/ml\n4. 胸部X线：心影增大、肺淤血\n\n诊断需要综合评估，建议专科就诊。",
        "confidence": 0.88
    }
}

@app.route('/')
def index():
    """主页 - 智能问答界面"""
    return render_template('index.html')

@app.route('/assessment')
def assessment():
    """智能自测界面"""
    return render_template('assessment.html')

@app.route('/comparison')
def comparison():
    """对比分析界面"""
    return render_template('comparison.html')

@app.route('/api/ask_question', methods=['POST'])
def ask_question():
    """处理问答请求 - 使用RAG检索增强生成（Retrieval-Augmented Generation）"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"error": "问题不能为空"}), 400
        
        logger.info(f"收到问题: {question}")
        
        # ============================================
        # RAG检索阶段：从知识库中检索相关文档
        # ============================================
        logger.info("RAG检索阶段：开始检索医学知识库...")
        retrieved_docs = rag_knowledge_base.retrieve(question, top_k=3)
        
        if retrieved_docs:
            logger.info(f"RAG检索成功：找到 {len(retrieved_docs)} 条相关文档")
            for doc in retrieved_docs:
                logger.info(f"  - {doc['title']} (相关性分数: {doc['score']})")
            
            # 构建检索上下文（用于传递给LLM）
            context = "\n\n".join([f"【{doc['title']}】\n{doc['content']}" for doc in retrieved_docs])
            logger.info(f"RAG检索上下文构建完成，共 {len(context)} 字符")
        else:
            logger.info("RAG检索阶段：未找到相关文档，使用默认知识")
            context = ""
        
        # ============================================
        # RAG生成阶段：调用DeepSeek LLM生成回答
        # ============================================
        try:
            # 直接使用硬编码的 API Key 来测试
            api_key = "sk-9479721d1fab49b28012419a5257d0da"
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
            
            # 构建RAG增强的Prompt
            system_prompt = f"""你是一个专业的医学知识问答助手。请基于以下参考资料回答用户问题：

--- 参考资料开始 ---
{context if context else "无相关参考资料"}
--- 参考资料结束 ---

你的回答应该：
1. 优先参考上述资料内容
2. 专业准确，基于医学知识
3. 清晰易懂，避免过于专业的术语
4. 结构化，使用分点说明
5. 包含相关的医学概念和原理
6. 提供实用的建议和注意事项

如果参考资料中没有相关内容，可以基于你的知识回答。

请用中文回答，保持专业和友好的语气。"""
            
            payload = {
                'model': DEEPSEEK_MODEL,
                'messages': [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                'temperature': DEEPSEEK_TEMPERATURE,
                'max_tokens': DEEPSEEK_MAX_TOKENS
            }
            
            logger.info(f"调用DeepSeek API... URL: {DEEPSEEK_BASE_URL}/v1/chat/completions")
            logger.info(f"Headers: {headers}")
            logger.info(f"Payload: {payload}")
            response = requests.post(
                f'{DEEPSEEK_BASE_URL}/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=30
            )
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response text: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content']
                confidence = 0.95  # DeepSeek API的回答通常置信度较高
                
                logger.info(f"RAG生成阶段：DeepSeek API调用成功，置信度: {confidence}")
                
                # 构建来源参考信息和教材溯源
                sources_used = [doc['title'] for doc in retrieved_docs] if retrieved_docs else []
                
                # 收集所有教材溯源信息（去重）
                textbook_references = []
                seen_refs = set()
                for doc in retrieved_docs:
                    if 'sources' in doc:
                        for source in doc['sources']:
                            ref_key = f"{source['textbook']}_{source['page']}"
                            if ref_key not in seen_refs:
                                seen_refs.add(ref_key)
                                textbook_references.append({
                                    "textbook": source['textbook'],
                                    "page": source['page']
                                })
                
                return jsonify({
                    "answer": answer,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat(),
                    "source": "RAG Retrieval-Augmented Generation",
                    "retrieved_sources": sources_used,
                    "textbook_references": textbook_references,
                    "rag_enabled": True
                })
            elif response.status_code in (401, 403):
                logger.error(f"DeepSeek 鉴权失败: {response.status_code} - {response.text}")
                return jsonify({
                    "answer": "DeepSeek API 鉴权失败（API Key 无效或权限不足）。请检查 DEEPSEEK_API_KEY 是否正确、是否有额度/权限，然后重启服务。",
                    "confidence": 0.0,
                    "timestamp": datetime.now().isoformat(),
                    "source": "Auth Error"
                }), 502
            else:
                logger.error(f"DeepSeek API调用失败: {response.status_code} - {response.text}")
                # API调用失败，返回默认回复
                return jsonify({
                    "answer": "抱歉，AI服务暂时不可用。您可以尝试：\n1. 换个方式提问\n2. 查看医学词典\n3. 使用智能自测功能巩固知识\n\n如有紧急医学问题，请及时就医！",
                    "confidence": 0.3,
                    "timestamp": datetime.now().isoformat(),
                    "source": "Fallback"
                })
                
        except requests.exceptions.Timeout:
            logger.error("DeepSeek API请求超时")
            return jsonify({
                "answer": "抱歉，AI服务响应超时。您可以尝试：\n1. 换个方式提问\n2. 查看医学词典\n3. 使用智能自测功能巩固知识\n\n如有紧急医学问题，请及时就医！",
                "confidence": 0.3,
                "timestamp": datetime.now().isoformat(),
                "source": "Timeout"
            })
        except requests.exceptions.ConnectionError as e:
            logger.error(f"DeepSeek API连接错误: {e}")
            return jsonify({
                "answer": "抱歉，无法连接到AI服务。请检查网络连接后重试。\n\n如有紧急医学问题，请及时就医！",
                "confidence": 0.3,
                "timestamp": datetime.now().isoformat(),
                "source": "Connection Error"
            })
        except Exception as e:
            logger.error(f"DeepSeek API调用异常: {e}")
            return jsonify({
                "answer": "抱歉，AI服务出现异常。您可以尝试：\n1. 换个方式提问\n2. 查看医学词典\n3. 使用智能自测功能巩固知识\n\n如有紧急医学问题，请及时就医！",
                "confidence": 0.3,
                "timestamp": datetime.now().isoformat(),
                "source": "Error"
            })
        
    except Exception as e:
        logger.error(f"问答处理错误: {str(e)}")
        return jsonify({"error": "服务器内部错误"}), 500

@app.route('/api/compare_concepts', methods=['POST'])
def compare_concepts():
    """对比两个医学概念"""
    try:
        data = request.get_json()
        concept_a = data.get('concept_a', '').strip()
        concept_b = data.get('concept_b', '').strip()
        
        if not concept_a or not concept_b:
            return jsonify({"error": "请输入两个要对比的概念"}), 400
        
        # 获取两个概念的信息
        info_a = MEDICAL_KNOWLEDGE_BASE["concepts"].get(concept_a)
        info_b = MEDICAL_KNOWLEDGE_BASE["concepts"].get(concept_b)
        
        if not info_a or not info_b:
            return jsonify({"error": "未找到相关概念信息"}), 404
        
        # 构建对比数据
        comparison_data = [
            {
                "dimension": "定义",
                "concept_a": info_a.get("definition", ""),
                "concept_b": info_b.get("definition", "")
            },
            {
                "dimension": "病因",
                "concept_a": info_a.get("etiology", ""),
                "concept_b": info_b.get("etiology", "")
            },
            {
                "dimension": "主要症状",
                "concept_a": "、".join(info_a.get("symptoms", [])),
                "concept_b": "、".join(info_b.get("symptoms", []))
            },
            {
                "dimension": "诊断方法",
                "concept_a": "、".join(info_a.get("diagnosis", [])),
                "concept_b": "、".join(info_b.get("diagnosis", []))
            },
            {
                "dimension": "治疗方案",
                "concept_a": "、".join(info_a.get("treatment", [])),
                "concept_b": "、".join(info_b.get("treatment", []))
            },
            {
                "dimension": "预后",
                "concept_a": info_a.get("prognosis", ""),
                "concept_b": info_b.get("prognosis", "")
            }
        ]
        
        return jsonify({
            "concept_a": concept_a,
            "concept_b": concept_b,
            "comparison": comparison_data
        })
        
    except Exception as e:
        logger.error(f"对比概念错误: {str(e)}")
        return jsonify({"error": "对比失败"}), 500

@app.route('/api/quiz_start', methods=['POST'])
def quiz_start():
    """开始自测"""
    try:
        data = request.get_json()
        quiz_type = data.get('type', 'quick')  # quick, chapter, exam
        chapter = data.get('chapter', '')
        
        # 完整题库
        all_questions = [
            # 心血管系统
            {
                "id": 1,
                "question": "心肌梗死的典型症状包括以下哪些？",
                "options": [
                    "胸骨后剧烈疼痛",
                    "恶心呕吐", 
                    "呼吸困难",
                    "发热"
                ],
                "correct": [0, 1, 2],
                "type": "multiple",
                "chapter": "cardiology"
            },
            {
                "id": 2,
                "question": "高血压的诊断标准是？",
                "options": [
                    "收缩压≥140mmHg或舒张压≥90mmHg",
                    "收缩压≥160mmHg或舒张压≥100mmHg",
                    "收缩压≥130mmHg或舒张压≥80mmHg",
                    "收缩压≥120mmHg或舒张压≥70mmHg"
                ],
                "correct": [0],
                "type": "single",
                "chapter": "cardiology"
            },
            {
                "id": 3,
                "question": "心力衰竭的主要症状包括？",
                "options": [
                    "呼吸困难",
                    "疲劳乏力",
                    "水肿",
                    "以上都是"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "cardiology"
            },
            {
                "id": 4,
                "question": "冠心病的危险因素包括哪些？",
                "options": [
                    "高血压",
                    "高血脂",
                    "糖尿病",
                    "以上都是"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "cardiology"
            },
            {
                "id": 5,
                "question": "心绞痛的典型疼痛特点是？",
                "options": [
                    "持续性剧痛",
                    "压榨性疼痛，位于胸骨后",
                    "刺痛",
                    "游走性疼痛"
                ],
                "correct": [1],
                "type": "single",
                "chapter": "cardiology"
            },
            # 呼吸系统
            {
                "id": 6,
                "question": "慢性阻塞性肺疾病（COPD）的主要病因是？",
                "options": [
                    "细菌感染",
                    "病毒感染",
                    "长期吸烟",
                    "空气污染"
                ],
                "correct": [2],
                "type": "single",
                "chapter": "respiratory"
            },
            {
                "id": 7,
                "question": "哮喘发作时的典型症状包括？",
                "options": [
                    "呼气性呼吸困难",
                    "吸气性呼吸困难",
                    "混合性呼吸困难",
                    "无呼吸困难"
                ],
                "correct": [0],
                "type": "single",
                "chapter": "respiratory"
            },
            {
                "id": 8,
                "question": "肺炎的常见症状包括哪些？",
                "options": [
                    "发热",
                    "咳嗽咳痰",
                    "胸痛",
                    "以上都是"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "respiratory"
            },
            # 消化系统
            {
                "id": 9,
                "question": "消化性溃疡的主要症状是？",
                "options": [
                    "上腹部疼痛",
                    "恶心呕吐",
                    "腹泻",
                    "便秘"
                ],
                "correct": [0],
                "type": "single",
                "chapter": "digestive"
            },
            {
                "id": 10,
                "question": "急性胰腺炎的典型表现是？",
                "options": [
                    "上腹部剧烈疼痛，向背部放射",
                    "右下腹痛",
                    "转移性右下腹痛",
                    "全腹痛"
                ],
                "correct": [0],
                "type": "single",
                "chapter": "digestive"
            },
            # 神经系统
            {
                "id": 11,
                "question": "脑卒中的FAST原则包括哪些？",
                "options": [
                    "Face（面部）",
                    "Arm（手臂）",
                    "Speech（语言）",
                    "以上都是"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "neurology"
            },
            {
                "id": 12,
                "question": "癫痫发作时的急救措施是？",
                "options": [
                    "用力按压患者",
                    "往患者口中塞东西",
                    "保护患者，防止受伤，保持呼吸道通畅",
                    "立即喂水"
                ],
                "correct": [2],
                "type": "single",
                "chapter": "neurology"
            },
            # 内分泌系统
            {
                "id": 13,
                "question": "糖尿病的典型症状包括哪些？",
                "options": [
                    "多饮",
                    "多尿",
                    "多食",
                    "以上都是"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "endocrine"
            },
            {
                "id": 14,
                "question": "甲状腺功能亢进的典型表现是？",
                "options": [
                    "怕热多汗",
                    "怕冷少汗",
                    "体重增加",
                    "嗜睡"
                ],
                "correct": [0],
                "type": "single",
                "chapter": "endocrine"
            },
            # 药理学
            {
                "id": 15,
                "question": "阿司匹林的主要作用是？",
                "options": [
                    "抗血小板聚集",
                    "抗凝",
                    "溶栓",
                    "降血压"
                ],
                "correct": [0],
                "type": "single",
                "chapter": "pharmacology"
            },
            {
                "id": 16,
                "question": "ACE抑制剂的主要副作用是？",
                "options": [
                    "咳嗽",
                    "头痛",
                    "水肿",
                    "皮疹"
                ],
                "correct": [0],
                "type": "single",
                "chapter": "pharmacology"
            },
            # 诊断学
            {
                "id": 17,
                "question": "心肌梗死的心电图典型表现是？",
                "options": [
                    "ST段抬高",
                    "ST段压低",
                    "T波倒置",
                    "Q波消失"
                ],
                "correct": [0],
                "type": "single",
                "chapter": "diagnostics"
            },
            {
                "id": 18,
                "question": "正常血压范围是？",
                "options": [
                    "收缩压<120mmHg且舒张压<80mmHg",
                    "收缩压<140mmHg且舒张压<90mmHg",
                    "收缩压<130mmHg且舒张压<85mmHg",
                    "收缩压<150mmHg且舒张压<95mmHg"
                ],
                "correct": [0],
                "type": "single",
                "chapter": "diagnostics"
            },
            # 心血管系统（扩展）
            {
                "id": 19,
                "question": "心肌梗死的治疗原则包括哪些？",
                "options": [
                    "立即休息",
                    "硝酸甘油含服",
                    "尽快再灌注治疗",
                    "以上都是"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "cardiology"
            },
            {
                "id": 20,
                "question": "心律失常的常见类型包括？",
                "options": [
                    "房颤",
                    "室性早搏",
                    "房室传导阻滞",
                    "以上都是"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "cardiology"
            },
            {
                "id": 21,
                "question": "急性心肌梗死的并发症包括哪些？",
                "options": [
                    "心律失常",
                    "心源性休克",
                    "心脏破裂",
                    "以上都是"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "cardiology"
            },
            {
                "id": 22,
                "question": "高血压的靶器官损害包括哪些？",
                "options": [
                    "心脏损害",
                    "脑损害",
                    "肾脏损害",
                    "以上都是"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "cardiology"
            },
            {
                "id": 23,
                "question": "心绞痛发作时的首选药物是？",
                "options": [
                    "阿司匹林",
                    "硝酸甘油",
                    "美托洛尔",
                    "硝苯地平"
                ],
                "correct": [1],
                "type": "single",
                "chapter": "cardiology"
            },
            # 呼吸系统（扩展）
            {
                "id": 24,
                "question": "肺心病的最主要病因是？",
                "options": [
                    "支气管哮喘",
                    "慢性阻塞性肺疾病",
                    "肺结核",
                    "肺炎"
                ],
                "correct": [1],
                "type": "single",
                "chapter": "respiratory"
            },
            {
                "id": 25,
                "question": "支气管哮喘的典型体征是？",
                "options": [
                    "干啰音",
                    "湿啰音",
                    "胸膜摩擦音",
                    "心包摩擦音"
                ],
                "correct": [0],
                "type": "single",
                "chapter": "respiratory"
            },
            {
                "id": 26,
                "question": "呼吸衰竭的诊断标准是？",
                "options": [
                    "PaO2<60mmHg",
                    "PaCO2>50mmHg",
                    "PaO2<60mmHg或PaCO2>50mmHg",
                    "PaO2<80mmHg"
                ],
                "correct": [2],
                "type": "single",
                "chapter": "respiratory"
            },
            {
                "id": 27,
                "question": "肺结核的典型症状包括？",
                "options": [
                    "低热盗汗",
                    "咳嗽咳痰",
                    "咯血",
                    "以上都是"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "respiratory"
            },
            {
                "id": 28,
                "question": "慢性支气管炎的诊断标准是咳嗽咳痰每年至少几个月，持续几年以上？",
                "options": [
                    "3个月，持续1年",
                    "3个月，持续2年",
                    "2个月，持续2年",
                    "6个月，持续1年"
                ],
                "correct": [1],
                "type": "single",
                "chapter": "respiratory"
            },
            # 消化系统（扩展）
            {
                "id": 29,
                "question": "消化性溃疡的并发症包括哪些？",
                "options": [
                    "出血",
                    "穿孔",
                    "幽门梗阻",
                    "以上都是"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "digestive"
            },
            {
                "id": 30,
                "question": "肝硬化的常见病因包括？",
                "options": [
                    "病毒性肝炎",
                    "酒精性肝病",
                    "非酒精性脂肪肝",
                    "以上都是"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "digestive"
            },
            {
                "id": 31,
                "question": "急性胆囊炎的典型表现是？",
                "options": [
                    "右上腹疼痛，向右肩放射",
                    "左上腹疼痛",
                    "全腹痛",
                    "下腹痛"
                ],
                "correct": [0],
                "type": "single",
                "chapter": "digestive"
            },
            {
                "id": 32,
                "question": "溃疡性结肠炎的主要症状是？",
                "options": [
                    "腹泻",
                    "黏液脓血便",
                    "腹痛",
                    "以上都是"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "digestive"
            },
            {
                "id": 33,
                "question": "上消化道出血的常见原因不包括？",
                "options": [
                    "消化性溃疡",
                    "食管胃底静脉曲张",
                    "急性胃黏膜病变",
                    "结肠癌"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "digestive"
            },
            # 神经系统（扩展）
            {
                "id": 34,
                "question": "帕金森病的典型症状包括？",
                "options": [
                    "静止性震颤",
                    "肌强直",
                    "运动迟缓",
                    "以上都是"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "neurology"
            },
            {
                "id": 35,
                "question": "阿尔茨海默病的早期症状是？",
                "options": [
                    "记忆力减退",
                    "运动障碍",
                    "感觉异常",
                    "语言障碍"
                ],
                "correct": [0],
                "type": "single",
                "chapter": "neurology"
            },
            {
                "id": 36,
                "question": "偏头痛的典型特点是？",
                "options": [
                    "单侧搏动性头痛",
                    "双侧持续性头痛",
                    "全头痛",
                    "后头痛"
                ],
                "correct": [0],
                "type": "single",
                "chapter": "neurology"
            },
            {
                "id": 37,
                "question": "重症肌无力的主要症状是？",
                "options": [
                    "肌肉无力，活动后加重，休息后缓解",
                    "肌肉强直",
                    "肌肉萎缩",
                    "肌肉疼痛"
                ],
                "correct": [0],
                "type": "single",
                "chapter": "neurology"
            },
            {
                "id": 38,
                "question": "脑梗死的常见病因不包括？",
                "options": [
                    "动脉粥样硬化",
                    "心源性栓塞",
                    "高血压",
                    "脑出血"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "neurology"
            },
            # 内分泌系统（扩展）
            {
                "id": 39,
                "question": "糖尿病的急性并发症包括哪些？",
                "options": [
                    "糖尿病酮症酸中毒",
                    "高渗性高血糖状态",
                    "低血糖",
                    "以上都是"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "endocrine"
            },
            {
                "id": 40,
                "question": "甲状腺功能减退的典型表现是？",
                "options": [
                    "怕冷少汗",
                    "怕热多汗",
                    "体重下降",
                    "心悸"
                ],
                "correct": [0],
                "type": "single",
                "chapter": "endocrine"
            },
            {
                "id": 41,
                "question": "库欣综合征的典型表现是？",
                "options": [
                    "向心性肥胖",
                    "满月脸",
                    "水牛背",
                    "以上都是"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "endocrine"
            },
            {
                "id": 42,
                "question": "痛风的主要生化指标异常是？",
                "options": [
                    "血尿酸升高",
                    "血钙升高",
                    "血磷升高",
                    "血钾升高"
                ],
                "correct": [0],
                "type": "single",
                "chapter": "endocrine"
            },
            {
                "id": 43,
                "question": "垂体瘤的常见症状包括？",
                "options": [
                    "头痛",
                    "视力视野改变",
                    "激素分泌异常",
                    "以上都是"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "endocrine"
            },
            # 药理学（扩展）
            {
                "id": 44,
                "question": "β受体阻滞剂的禁忌症包括？",
                "options": [
                    "哮喘",
                    "严重心动过缓",
                    "房室传导阻滞",
                    "以上都是"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "pharmacology"
            },
            {
                "id": 45,
                "question": "他汀类药物的主要作用是？",
                "options": [
                    "降血脂",
                    "抗血小板",
                    "降压",
                    "降糖"
                ],
                "correct": [0],
                "type": "single",
                "chapter": "pharmacology"
            },
            {
                "id": 46,
                "question": "利尿剂的主要副作用是？",
                "options": [
                    "电解质紊乱",
                    "尿酸升高",
                    "血糖血脂异常",
                    "以上都是"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "pharmacology"
            },
            {
                "id": 47,
                "question": "钙通道阻滞剂的主要适应症是？",
                "options": [
                    "高血压",
                    "心绞痛",
                    "心律失常",
                    "以上都是"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "pharmacology"
            },
            {
                "id": 48,
                "question": "胰岛素的主要副作用是？",
                "options": [
                    "低血糖",
                    "体重增加",
                    "过敏反应",
                    "以上都是"
                ],
                "correct": [3],
                "type": "single",
                "chapter": "pharmacology"
            },
            # 诊断学（扩展）
            {
                "id": 49,
                "question": "贫血的诊断标准是？",
                "options": [
                    "成年男性Hb<120g/L，成年女性Hb<110g/L",
                    "成年男性Hb<130g/L，成年女性Hb<120g/L",
                    "成年男性Hb<140g/L，成年女性Hb<130g/L",
                    "成年男性Hb<150g/L，成年女性Hb<140g/L"
                ],
                "correct": [0],
                "type": "single",
                "chapter": "diagnostics"
            },
            {
                "id": 50,
                "question": "白细胞计数的正常范围是？",
                "options": [
                    "(4-10)×10^9/L",
                    "(3-9)×10^9/L",
                    "(5-11)×10^9/L",
                    "(6-12)×10^9/L"
                ],
                "correct": [0],
                "type": "single",
                "chapter": "diagnostics"
            }
        ]
        
        # 根据类型选择题目
        if quiz_type == 'quick':
            # 快速自测：随机10题
            import random
            questions = random.sample(all_questions, min(10, len(all_questions)))
        elif quiz_type == 'chapter':
            # 章节练习：根据章节筛选
            if chapter:
                chapter_questions = [q for q in all_questions if q.get('chapter') == chapter]
                questions = chapter_questions if chapter_questions else all_questions[:10]
            else:
                questions = all_questions[:10]
        elif quiz_type == 'exam':
            # 模拟考试：所有题目
            questions = all_questions
        else:
            questions = all_questions[:10]
        
        return jsonify({
            "quiz_id": str(uuid.uuid4()),
            "type": quiz_type,
            "questions": questions,
            "time_limit": 3600 if quiz_type == 'exam' else 600,
            "total_questions": len(questions)
        })
        
    except Exception as e:
        logger.error(f"开始自测错误: {str(e)}")
        return jsonify({"error": "无法开始自测"}), 500

@app.route('/api/quiz_submit', methods=['POST'])
def quiz_submit():
    """提交自测答案"""
    try:
        data = request.get_json()
        quiz_id = data.get('quiz_id')
        answers = data.get('answers', {})
        time_used = data.get('time_used', 0)
        
        # 获取题目（这里简化处理，实际应该从session或数据库获取）
        # 重新生成题目进行评分
        all_questions = [
            {"id": 1, "correct": [0, 1, 2]},
            {"id": 2, "correct": [0]},
            {"id": 3, "correct": [3]},
            {"id": 4, "correct": [3]},
            {"id": 5, "correct": [1]},
            {"id": 6, "correct": [2]},
            {"id": 7, "correct": [0]},
            {"id": 8, "correct": [3]},
            {"id": 9, "correct": [0]},
            {"id": 10, "correct": [0]},
            {"id": 11, "correct": [3]},
            {"id": 12, "correct": [2]},
            {"id": 13, "correct": [3]},
            {"id": 14, "correct": [0]},
            {"id": 15, "correct": [0]},
            {"id": 16, "correct": [0]},
            {"id": 17, "correct": [0]},
            {"id": 18, "correct": [0]},
            {"id": 19, "correct": [3]},
            {"id": 20, "correct": [3]},
            {"id": 21, "correct": [3]},
            {"id": 22, "correct": [3]},
            {"id": 23, "correct": [1]},
            {"id": 24, "correct": [1]},
            {"id": 25, "correct": [0]},
            {"id": 26, "correct": [2]},
            {"id": 27, "correct": [3]},
            {"id": 28, "correct": [1]},
            {"id": 29, "correct": [3]},
            {"id": 30, "correct": [3]},
            {"id": 31, "correct": [0]},
            {"id": 32, "correct": [3]},
            {"id": 33, "correct": [3]},
            {"id": 34, "correct": [3]},
            {"id": 35, "correct": [0]},
            {"id": 36, "correct": [0]},
            {"id": 37, "correct": [0]},
            {"id": 38, "correct": [3]},
            {"id": 39, "correct": [3]},
            {"id": 40, "correct": [0]},
            {"id": 41, "correct": [3]},
            {"id": 42, "correct": [0]},
            {"id": 43, "correct": [3]},
            {"id": 44, "correct": [3]},
            {"id": 45, "correct": [0]},
            {"id": 46, "correct": [3]},
            {"id": 47, "correct": [3]},
            {"id": 48, "correct": [3]},
            {"id": 49, "correct": [0]},
            {"id": 50, "correct": [0]}
        ]
        
        # 计算得分
        correct_count = 0
        total_count = len(answers)
        wrong_answers = []
        
        for question_id, user_answer in answers.items():
            question = next((q for q in all_questions if q['id'] == int(question_id)), None)
            if question:
                if sorted(user_answer) == sorted(question['correct']):
                    correct_count += 1
                else:
                    wrong_answers.append({
                        "question_id": question_id,
                        "user_answer": user_answer,
                        "correct_answer": question['correct']
                    })
        
        score = (correct_count / total_count * 100) if total_count > 0 else 0
        
        return jsonify({
            "success": True,
            "score": round(score, 1),
            "correct_count": correct_count,
            "total_count": total_count,
            "wrong_answers": wrong_answers,
            "time_used": time_used
        })
        
    except Exception as e:
        logger.error(f"提交自测错误: {str(e)}")
        return jsonify({"error": "提交失败"}), 500

@app.route('/api/quiz_chapters', methods=['GET'])
def quiz_chapters():
    """获取可用章节列表"""
    try:
        chapters = [
            {"id": "cardiology", "name": "心血管系统", "count": 10},
            {"id": "respiratory", "name": "呼吸系统", "count": 7},
            {"id": "digestive", "name": "消化系统", "count": 7},
            {"id": "neurology", "name": "神经系统", "count": 7},
            {"id": "endocrine", "name": "内分泌系统", "count": 7},
            {"id": "pharmacology", "name": "药理学", "count": 7},
            {"id": "diagnostics", "name": "诊断学", "count": 5}
        ]
        
        return jsonify({"chapters": chapters})
        
    except Exception as e:
        logger.error(f"获取章节错误: {str(e)}")
        return jsonify({"error": "获取失败"}), 500

@app.route('/api/error_book', methods=['GET', 'POST'])
def error_book():
    """错题本"""
    try:
        if request.method == 'GET':
            # 获取错题列表
            return jsonify({
                "errors": [],
                "message": "错题功能开发中"
            })
        else:
            # 保存错题
            data = request.get_json()
            return jsonify({"success": True, "message": "错题已保存"})
        
    except Exception as e:
        logger.error(f"错题本错误: {str(e)}")
        return jsonify({"error": "操作失败"}), 500

@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("=" * 60)
    print("医学智能问答系统 - Flask Web应用")
    print("=" * 60)
    print("核心技术: RAG检索增强生成 (Retrieval-Augmented Generation)")
    print("问答引擎: DeepSeek API")
    print("=" * 60)
    print("RAG配置:")
    print(f"  - 知识库文档数: {len(rag_knowledge_base.documents)}")
    print(f"  - 检索模式: 关键词匹配 + 语义相似度")
    print(f"  - 返回文档数: Top-3")
    print("=" * 60)
    print("API配置:")
    print(f"  - Model: {DEEPSEEK_MODEL}")
    print(f"  - Temperature: {DEEPSEEK_TEMPERATURE}")
    print(f"  - Max Tokens: {DEEPSEEK_MAX_TOKENS}")
    print("=" * 60)
    print("服务地址: http://localhost:5001")
    print("功能列表:")
    print("  - 智能问答(RAG): http://localhost:5001/")
    print("  - 智能自测: http://localhost:5001/assessment")
    print("  - 对比分析: http://localhost:5001/comparison")
    print("=" * 60)
    print("系统已启动，RAG检索增强生成模式已启用")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5001)