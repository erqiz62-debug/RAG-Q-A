#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek智能问答系统 - Web版本
提供Web界面和API接口
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from flask import Flask, request, jsonify, render_template_string

# 添加scripts目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from local_vector_database import LocalVectorDatabase
from deepseek_medical_qa import DeepSeekMedicalQA

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 初始化问答系统
qa_system = None


def init_qa_system():
    """初始化问答系统"""
    global qa_system
    try:
        qa_system = DeepSeekMedicalQA(
            vector_db_dir="./data/chroma_db",
            collection_name="medical_knowledge"
        )
        logger.info("DeepSeek问答系统初始化成功")
    except Exception as e:
        logger.error(f"初始化问答系统失败: {e}")
        qa_system = None


# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepSeek智能医学问答系统</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            text-align: center;
        }
        
        .header h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        
        .input-section, .output-section {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .section-title {
            color: #667eea;
            font-size: 1.5em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        .input-group label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: bold;
        }
        
        .input-group input, 
        .input-group textarea,
        .input-group select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        
        .input-group input:focus,
        .input-group textarea:focus,
        .input-group select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .input-group textarea {
            min-height: 120px;
            resize: vertical;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            cursor: pointer;
            transition: transform 0.3s, box-shadow 0.3s;
            width: 100%;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .result-box {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin-bottom: 20px;
        }
        
        .result-box h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .result-box .answer {
            color: #333;
            line-height: 1.8;
            margin-bottom: 15px;
        }
        
        .result-box .meta-info {
            color: #666;
            font-size: 0.9em;
            padding-top: 10px;
            border-top: 1px solid #e0e0e0;
        }
        
        .result-box .meta-info div {
            margin-bottom: 5px;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #667eea;
        }
        
        .loading::after {
            content: '...';
            animation: dots 1.5s steps(4, end) infinite;
        }
        
        @keyframes dots {
            0%, 20% { color: rgba(0,0,0,0); text-shadow: .25em 0 0 rgba(0,0,0,0), .5em 0 0 rgba(0,0,0,0);}
            40% { color: white; text-shadow: .25em 0 0 rgba(0,0,0,0), .5em 0 0 rgba(0,0,0,0);}
            60% { text-shadow: .25em 0 0 white, .5em 0 0 rgba(0,0,0,0);}
            80%, 100% { text-shadow: .25em 0 0 white, .5em 0 0 white;}
        }
        
        .error {
            background: #fee;
            border-left-color: #f44;
            color: #c33;
        }
        
        .status-bar {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-top: 30px;
        }
        
        .status-item {
            display: inline-block;
            margin-right: 30px;
            margin-bottom: 10px;
        }
        
        .status-item strong {
            color: #667eea;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 DeepSeek智能医学问答系统</h1>
            <p>基于DeepSeek大模型的智能医学问答服务</p>
        </div>
        
        <div class="main-content">
            <div class="input-section">
                <h2 class="section-title">提问</h2>
                
                <div class="input-group">
                    <label for="question">您的问题：</label>
                    <textarea id="question" placeholder="请输入您的医学问题..."></textarea>
                </div>
                
                <div class="input-group">
                    <label for="useKnowledge">使用知识库检索：</label>
                    <select id="useKnowledge">
                        <option value="true">是</option>
                        <option value="false">否</option>
                    </select>
                </div>
                
                <div class="input-group">
                    <label for="topK">检索结果数量：</label>
                    <select id="topK">
                        <option value="3">3</option>
                        <option value="5">5</option>
                        <option value="10">10</option>
                    </select>
                </div>
                
                <button class="btn" onclick="askQuestion()" id="askBtn">提交问题</button>
            </div>
            
            <div class="output-section">
                <h2 class="section-title">回答</h2>
                <div id="resultArea">
                    <div style="text-align: center; color: #999; padding: 40px;">
                        请在左侧输入您的问题，然后点击"提交问题"按钮
                    </div>
                </div>
            </div>
        </div>
        
        <div class="status-bar">
            <h3 style="color: #667eea; margin-bottom: 15px;">系统状态</h3>
            <div class="status-item">
                <strong>问答引擎:</strong> DeepSeek API
            </div>
            <div class="status-item">
                <strong>模型:</strong> {{ model_name }}
            </div>
            <div class="status-item">
                <strong>知识库条目:</strong> {{ knowledge_count }}
            </div>
            <div class="status-item">
                <strong>系统时间:</strong> {{ current_time }}
            </div>
        </div>
    </div>
    
    <script>
        function askQuestion() {
            const question = document.getElementById('question').value.trim();
            const useKnowledge = document.getElementById('useKnowledge').value === 'true';
            const topK = parseInt(document.getElementById('topK').value);
            
            if (!question) {
                alert('请输入您的问题！');
                return;
            }
            
            const resultArea = document.getElementById('resultArea');
            const askBtn = document.getElementById('askBtn');
            
            // 显示加载状态
            resultArea.innerHTML = '<div class="loading">正在思考</div>';
            askBtn.disabled = true;
            
            // 发送请求
            fetch('/api/answer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    use_knowledge: useKnowledge,
                    top_k: topK
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    displayResult(data.result);
                } else {
                    displayError(data.error || '回答失败');
                }
            })
            .catch(error => {
                displayError('网络错误: ' + error.message);
            })
            .finally(() => {
                askBtn.disabled = false;
            });
        }
        
        function displayResult(result) {
            const resultArea = document.getElementById('resultArea');
            
            let sourcesHtml = '';
            if (result['知识来源'] && result['知识来源'].length > 0) {
                sourcesHtml = '<div class="meta-info"><strong>知识来源:</strong><br>';
                result['知识来源'].forEach((source, index) => {
                    sourcesHtml += `<div>• ${source['document']} (第${source['page']}页, 分数: ${source['score'].toFixed(2)})</div>`;
                });
                sourcesHtml += '</div>';
            }
            
            resultArea.innerHTML = `
                <div class="result-box">
                    <h3>问题</h3>
                    <div style="margin-bottom: 15px; color: #333;">${result['问题']}</div>
                    
                    <h3>答案</h3>
                    <div class="answer">${formatAnswer(result['答案'])}</div>
                    
                    <div class="meta-info">
                        <div><strong>置信度:</strong> ${(result['置信度'] * 100).toFixed(1)}%</div>
                        <div><strong>使用知识库:</strong> ${result['使用知识库'] ? '是' : '否'}</div>
                        <div><strong>检索到的知识数量:</strong> ${result['检索到的知识数量']}</div>
                        <div><strong>回答时间:</strong> ${result['回答时间']}</div>
                        ${sourcesHtml}
                    </div>
                </div>
            `;
        }
        
        function formatAnswer(answer) {
            // 简单的格式化，将换行符转换为<br>
            return answer.replace(/\n/g, '<br>');
        }
        
        function displayError(message) {
            const resultArea = document.getElementById('resultArea');
            resultArea.innerHTML = `
                <div class="result-box error">
                    <h3>错误</h3>
                    <div class="answer">${message}</div>
                </div>
            `;
        }
        
        // 页面加载时获取系统状态
        window.onload = function() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.querySelector('.status-bar').innerHTML = `
                            <h3 style="color: #667eea; margin-bottom: 15px;">系统状态</h3>
                            <div class="status-item">
                                <strong>问答引擎:</strong> ${data.result['问答引擎']}
                            </div>
                            <div class="status-item">
                                <strong>模型:</strong> ${data.result['API配置']['模型']}
                            </div>
                            <div class="status-item">
                                <strong>知识库条目:</strong> ${data.result['知识库条目']}
                            </div>
                            <div class="status-item">
                                <strong>倒排索引大小:</strong> ${data.result['倒排索引大小']}
                            </div>
                            <div class="status-item">
                                <strong>系统时间:</strong> ${data.result['最后更新']}
                            </div>
                        `;
                    }
                });
        };
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """首页"""
    if qa_system:
        status = qa_system.get_system_status()
        return render_template_string(
            HTML_TEMPLATE,
            model_name=status['API配置']['模型'],
            knowledge_count=status['知识库条目'],
            current_time=status['最后更新']
        )
    else:
        return render_template_string(
            HTML_TEMPLATE,
            model_name="未初始化",
            knowledge_count=0,
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )


@app.route('/api/status')
def get_status():
    """获取系统状态API"""
    if qa_system:
        try:
            status = qa_system.get_system_status()
            return jsonify({
                'success': True,
                'result': status
            })
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            })
    else:
        return jsonify({
            'success': False,
            'error': '问答系统未初始化'
        })


@app.route('/api/answer', methods=['POST'])
def answer_question():
    """问答API"""
    if not qa_system:
        return jsonify({
            'success': False,
            'error': '问答系统未初始化'
        }), 500
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        use_knowledge = data.get('use_knowledge', True)
        top_k = data.get('top_k', 3)
        
        if not question:
            return jsonify({
                'success': False,
                'error': '问题不能为空'
            }), 400
        
        logger.info(f"收到问题: {question}")
        logger.info(f"参数: use_knowledge={use_knowledge}, top_k={top_k}")
        
        # 调用问答系统
        result = qa_system.answer_question(
            question=question,
            use_knowledge=use_knowledge,
            top_k=top_k
        )
        
        logger.info(f"回答完成，置信度: {result['置信度']}")
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"处理问题时出错: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/batch_answer', methods=['POST'])
def batch_answer_questions():
    """批量问答API"""
    if not qa_system:
        return jsonify({
            'success': False,
            'error': '问答系统未初始化'
        }), 500
    
    try:
        data = request.get_json()
        questions = data.get('questions', [])
        
        if not questions:
            return jsonify({
                'success': False,
                'error': '问题列表不能为空'
            }), 400
        
        logger.info(f"收到批量问题: {len(questions)} 个")
        
        # 调用批量问答
        results = qa_system.batch_answer_questions(questions)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"批量回答问题时出错: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy' if qa_system else 'unhealthy',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


def main():
    """主函数"""
    print("=" * 60)
    print("DeepSeek智能医学问答系统 - Web版本")
    print("=" * 60)
    print()
    
    # 初始化问答系统
    init_qa_system()
    
    if not qa_system:
        print("❌ 问答系统初始化失败，请检查配置")
        return
    
    # 显示系统状态
    status = qa_system.get_system_status()
    print("系统状态:")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print()
    
    # 启动Flask应用
    host = '0.0.0.0'
    port = 5001
    
    print(f"🚀 启动Web服务...")
    print(f"📱 访问地址: http://localhost:{port}")
    print(f"🌐 外部访问: http://<your-ip>:{port}")
    print(f"📊 API状态: http://localhost:{port}/api/status")
    print(f"❤️  健康检查: http://localhost:{port}/health")
    print()
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    print()
    
    try:
        app.run(host=host, port=port, debug=False)
    except KeyboardInterrupt:
        print("\n服务已停止")


if __name__ == "__main__":
    main()
