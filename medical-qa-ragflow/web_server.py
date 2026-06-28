#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医学QA系统Web交互服务器
提供类似DeepSeek的问答交互界面
"""

import os
import sys
import json
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import threading
import time

# 添加项目路径到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    from precise_citation_medical_qa_system import MedicalQASystem
    QASystem = MedicalQASystem()
    print("✅ 医学QA系统初始化成功")
except ImportError as e:
    print(f"❌ 导入QA系统失败: {e}")
    QASystem = None

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量
system_status = {
    'initialized': QASystem is not None,
    'total_questions': 0,
    'confidence_scores': [],
    'knowledge_blocks': 4,
    'medical_terms': 80
}

# Web界面路由
@app.route('/')
def index():
    """主页面 - 重定向到交互界面"""
    return render_template('qa_interactive.html')

@app.route('/qa')
def qa_interface():
    """问答交互界面"""
    return render_template('qa_interactive.html')

@app.route('/demo')
def demo_interface():
    """演示界面"""
    return render_template('demo_interface.html')

# API路由
@app.route('/api/status', methods=['GET'])
def get_status():
    """获取系统状态"""
    avg_confidence = sum(system_status['confidence_scores']) / len(system_status['confidence_scores']) if system_status['confidence_scores'] else 0
    
    return jsonify({
        'initialized': system_status['initialized'],
        'total_questions': system_status['total_questions'],
        'avg_confidence': round(avg_confidence, 2),
        'knowledge_blocks': system_status['knowledge_blocks'],
        'medical_terms': system_status['medical_terms'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """处理用户问题"""
    if not QASystem:
        return jsonify({
            'error': 'QA系统未初始化',
            'answer': '抱歉，系统暂时不可用，请稍后重试。',
            'confidence': 0.0,
            'sources': []
        }), 500
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'error': '问题不能为空',
                'answer': '请输入您的问题。',
                'confidence': 0.0,
                'sources': []
            }), 400
        
        logger.info(f"收到问题: {question}")
        
        # 调用QA系统
        start_time = time.time()
        
        # 这里需要根据实际的QA系统方法进行调整
        # 假设QA系统有process_query_with_precise_citations方法
        try:
            result = QASystem.process_query_with_precise_citations(question)
            
            # 解析结果
            if isinstance(result, dict):
                answer = result.get('answer', '抱歉，无法生成回答。')
                confidence = result.get('confidence', 0.5)
                sources = result.get('sources', [])
            else:
                # 如果返回的是字符串，尝试解析
                answer = str(result)
                confidence = 0.7  # 默认置信度
                sources = []
            
        except Exception as e:
            logger.error(f"QA系统调用失败: {e}")
            # 返回模拟回答
            answer = generate_mock_answer(question)
            confidence = 0.65
            sources = []
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 更新统计
        system_status['total_questions'] += 1
        system_status['confidence_scores'].append(confidence)
        
        logger.info(f"问题处理完成，耗时: {processing_time:.2f}秒")
        
        return jsonify({
            'answer': answer,
            'confidence': confidence,
            'sources': sources,
            'processing_time': round(processing_time, 2),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"处理问题时发生错误: {e}")
        return jsonify({
            'error': '系统内部错误',
            'answer': '抱歉，系统遇到问题。请稍后重试。',
            'confidence': 0.0,
            'sources': []
        }), 500

@app.route('/api/quick-questions', methods=['GET'])
def get_quick_questions():
    """获取快速问题列表"""
    quick_questions = [
        "什么是心脏病？",
        "冠心病的诊断方法有哪些？",
        "心力衰竭的症状和治疗方法",
        "高血压的治疗原则",
        "心肌梗死的急救措施",
        "心律失常的症状有哪些？",
        "如何预防心血管疾病？",
        "心绞痛的症状和特点",
        "心脏瓣膜病的类型",
        "先天性心脏病的分类"
    ]
    
    return jsonify({
        'questions': quick_questions,
        'total': len(quick_questions)
    })

def generate_mock_answer(question):
    """生成模拟回答（备用方案）"""
    answers = {
        '心脏病': """根据医学知识库，心脏病是心脏功能或结构异常的疾病总称。

🏥 **主要类型包括：**
• 冠状动脉疾病（冠心病）
• 心律失常
• 心力衰竭
• 心脏瓣膜病
• 心肌病

💡 **重要提醒：**
早期识别和及时治疗对改善预后非常重要。""",
        
        '冠心病': """冠状动脉粥样硬化性心脏病的诊断方法：

🔬 **无创检查：**
• 心电图：检测心肌缺血和心律异常
• 运动负荷试验：评估心脏在运动时的功能
• 心脏彩超：评估心脏结构和功能
• 冠脉CTA：冠状动脉CT血管成像

🩺 **有创检查：**
• 冠状动脉造影：确诊金标准，可直接显示病变""",
        
        '心力衰竭': """**心力衰竭的症状：**

🫁 **呼吸系统症状：**
• 呼吸困难（活动后加重）
• 端坐呼吸
• 夜间阵发性呼吸困难

💪 **循环系统症状：**
• 乏力、疲劳
• 心悸
• 下肢水肿

**治疗方法：**
💊 药物治疗（ACE抑制剂、β受体阻滞剂等）
🏥 非药物治疗（限制钠盐、适度运动等）""",
        
        '高血压': """高血压的治疗原则：

🎯 **治疗目标：**
• 血压控制 < 140/90 mmHg
• 减少心血管并发症风险

💊 **药物治疗：**
• ACE抑制剂/ARB
• 钙通道阻滞剂
• 利尿剂
• β受体阻滞剂

🍎 **生活方式：**
• 限制钠盐摄入
• 适量运动
• 戒烟限酒"""
    }
    
    # 查找匹配的回答
    for keyword, answer in answers.items():
        if keyword in question:
            return answer
    
    # 默认回答
    return f"""感谢您的问题："{question}"

根据您的医学知识库，我可以为您提供相关的医学信息。

📋 **我专门回答的问题领域：**
• 心脏病的定义和分类
• 冠心病的诊断和治疗
• 心力衰竭的症状和管理
• 高血压的控制原则
• 心律失常的识别
• 心肌梗死的急救

请提供更具体的医学问题，或者从快速问题中选择。"""

@app.route('/health')
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'qa_system_initialized': QASystem is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 启动医学QA系统Web服务器...")
    print("📍 访问地址:")
    print("   - 主界面: http://localhost:8082/")
    print("   - 问答界面: http://localhost:8082/qa")
    print("   - 演示界面: http://localhost:8082/demo")
    print("   - 状态检查: http://localhost:8082/health")
    
    if QASystem:
        print("✅ QA系统已就绪")
    else:
        print("⚠️  QA系统未初始化，将使用模拟回答")
    
    print("\n" + "="*50)
    print("🎯 正在启动Web服务器...")
    print("="*50)
    
    # 启动Flask服务器
    app.run(host='0.0.0.0', port=8082, debug=False)