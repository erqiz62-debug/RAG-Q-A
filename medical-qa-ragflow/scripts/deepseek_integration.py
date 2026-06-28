#!/usr/bin/env python3
"""
DeepSeek大模型集成模块
为RAGFlow医学问答系统提供DeepSeek API接口
"""

import os
import json
import logging
import requests
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MedicalQAConfig:
    """医学问答配置"""
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    max_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.9
    citation_format: str = "page_reference"
    response_language: str = "chinese"

class DeepSeekClient:
    """DeepSeek API客户端"""
    
    def __init__(self, config: MedicalQAConfig):
        self.config = config
        self.api_url = f"{config.deepseek_base_url}/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {config.deepseek_api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, prompt: str, context: List[Dict]) -> Dict[str, Any]:
        """生成医学问答回复"""
        try:
            # 构建完整的提示词
            full_prompt = self._build_medical_prompt(prompt, context)
            
            payload = {
                "model": self.config.deepseek_model,
                "messages": [
                    {"role": "system", "content": full_prompt["system"]},
                    {"role": "user", "content": full_prompt["user"]}
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p
            }
            
            logger.info("发送请求到DeepSeek API...")
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                assistant_message = result["choices"][0]["message"]["content"]
                
                # 解析回复以提取引用信息
                parsed_response = self._parse_response(assistant_message, context)
                return {
                    "success": True,
                    "response": parsed_response["content"],
                    "citations": parsed_response["citations"],
                    "confidence_score": parsed_response["confidence"],
                    "source_count": len(context)
                }
            else:
                logger.error(f"DeepSeek API错误: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"API调用失败: {response.status_code}",
                    "response": "抱歉，系统暂时无法回答您的问题。",
                    "citations": []
                }
                
        except Exception as e:
            logger.error(f"生成回复时发生错误: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "抱歉，系统遇到了技术问题。",
                "citations": []
            }
    
    def _build_medical_prompt(self, question: str, context: List[Dict]) -> Dict[str, str]:
        """构建医学专业提示词"""
        
        # 构建上下文信息
        context_text = ""
        for i, ctx in enumerate(context):
            context_text += f"""
引用文献 {i+1}:
- 来源: {ctx.get('metadata', {}).get('document_title', '未知文档')}
- 页码: {ctx.get('metadata', {}).get('page_number', 'N/A')}
- 内容: {ctx.get('content', '')[:200]}...

"""
        
        system_prompt = f"""你是一位资深的医学专家和教授，具有丰富的临床经验和教学经验。你专门负责回答医学教材相关的问题。

## 核心要求：
1. **严格基于引用文献回答**：你的回答必须严格基于提供的引用文献内容，不得添加超出引用文献的信息。
2. **保持学术严谨性**：使用专业的医学术语，确保答案准确、专业、权威。
3. **保持逻辑清晰**：按照"病因-病理-临床表现-诊断-治疗"的逻辑结构组织答案。
4. **答案溯源**：在回答中明确标注引用的文献来源和页码。
5. **语言要求**：使用中文回答，语言专业、简洁明了。

## 回答格式要求：
- 首先给出直接回答
- 然后提供详细的解释和分析
- 最后提供相关的注意事项或建议
- 使用[引用X]格式标注引用来源

## 医学知识完整性：
确保你的回答体现医学知识的完整链条，如：
- 病因分析
- 病理机制
- 临床表现
- 诊断方法
- 治疗方案
- 预后评估

记住：你是一位医学教授，你的回答应该体现深厚的医学专业知识和教学经验。"""

        user_prompt = f"""基于以下引用文献，回答这个医学问题：

**问题**：{question}

**引用文献**：
{context_text}

**要求**：
1. 严格基于上述引用文献内容回答
2. 提供专业、准确的医学答案
3. 明确标注引用来源
4. 保持逻辑清晰和学术严谨性
5. 如引用文献不足以完整回答问题，请明确说明"""

        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def _parse_response(self, response: str, context: List[Dict]) -> Dict[str, Any]:
        """解析模型回复，提取引用信息"""
        lines = response.split('\n')
        citations = []
        content_lines = []
        
        for line in lines:
            # 检测引用标记 [引用X]
            import re
            citation_matches = re.findall(r'\[引用(\d+)\]', line)
            if citation_matches:
                for match in citation_matches:
                    idx = int(match) - 1
                    if 0 <= idx < len(context):
                        ctx = context[idx]
                        citations.append({
                            "citation_index": idx + 1,
                            "document_title": ctx.get('metadata', {}).get('document_title', '未知'),
                            "page_number": ctx.get('metadata', {}).get('page_number', 'N/A'),
                            "content_snippet": ctx.get('content', '')[:100] + "..."
                        })
            
            content_lines.append(line)
        
        # 计算置信度分数
        confidence_score = self._calculate_confidence_score(response, len(citations), len(context))
        
        return {
            "content": '\n'.join(content_lines),
            "citations": citations,
            "confidence": confidence_score
        }
    
    def _calculate_confidence_score(self, response: str, citation_count: int, context_count: int) -> float:
        """计算答案置信度分数"""
        if context_count == 0:
            return 0.0
        
        # 基于引用比例计算置信度
        citation_ratio = citation_count / context_count
        
        # 基于回答长度和详细程度
        response_length = len(response)
        length_score = min(response_length / 500, 1.0)  # 500字符为满分
        
        # 综合置信度
        confidence = (citation_ratio * 0.6 + length_score * 0.4)
        
        return min(confidence, 1.0)

class MedicalQASystem:
    """医学问答系统主类"""
    
    def __init__(self, config_path: str = "./configs/ragflow_config.json"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 初始化DeepSeek客户端（需要API密钥）
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            logger.warning("未设置DEEPSEEK_API_KEY环境变量")
            self.deepseek_client = None
        else:
            qa_config = MedicalQAConfig(
                deepseek_api_key=api_key,
                deepseek_base_url=self.config['ragflow']['models']['llm']['provider_config']['base_url']
                if 'provider_config' in self.config['ragflow']['models']['llm'] 
                else "https://api.deepseek.com/v1"
            )
            self.deepseek_client = DeepSeekClient(qa_config)
    
    def answer_question(self, question: str, retrieved_context: List[Dict]) -> Dict[str, Any]:
        """回答医学问题"""
        if not self.deepseek_client:
            return {
                "success": False,
                "error": "DeepSeek API未配置",
                "response": "请先配置DeepSeek API密钥",
                "citations": []
            }
        
        return self.deepseek_client.generate_response(question, retrieved_context)
    
    def create_fallback_response(self, question: str, context: List[Dict]) -> Dict[str, Any]:
        """创建备用回复（当API不可用时）"""
        if not context:
            return {
                "success": False,
                "response": "抱歉，未找到相关的医学资料来回答您的问题。",
                "citations": []
            }
        
        # 简单的基于检索的回复生成
        response_parts = []
        citations = []
        
        for i, ctx in enumerate(context[:3]):  # 最多使用3个上下文
            response_parts.append(f"根据《{ctx.get('metadata', {}).get('document_title', '医学教材')}》第{ctx.get('metadata', {}).get('page_number', 'N/A')}页：")
            response_parts.append(ctx.get('content', '')[:300] + "...")
            
            citations.append({
                "citation_index": i + 1,
                "document_title": ctx.get('metadata', {}).get('document_title', '未知'),
                "page_number": ctx.get('metadata', {}).get('page_number', 'N/A'),
                "content_snippet": ctx.get('content', '')[:100] + "..."
            })
        
        full_response = "基于检索到的医学资料：\n\n" + "\n\n".join(response_parts)
        
        return {
            "success": True,
            "response": full_response,
            "citations": citations,
            "confidence_score": 0.7,
            "source_count": len(context)
        }

def main():
    """测试函数"""
    # 创建测试配置
    test_config = MedicalQAConfig(
        deepseek_api_key="test_key"  # 替换为实际API密钥
    )
    
    # 测试上下文
    test_context = [
        {
            "content": "心脏疾病的病因主要包括先天性因素和后天性因素。先天性心脏病是由于胚胎期心脏发育异常所致。",
            "metadata": {
                "document_title": "心脏病学",
                "page_number": "15"
            }
        }
    ]
    
    # 测试问题
    test_question = "心脏疾病的病因是什么？"
    
    # 初始化系统
    qa_system = MedicalQASystem()
    
    # 测试回答
    result = qa_system.answer_question(test_question, test_context)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()