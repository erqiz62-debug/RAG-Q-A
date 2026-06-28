#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接启动DeepSeek智能问答系统
"""

import os
import sys

# 设置环境变量
os.environ['LLM_API_KEY'] = 'sk-0553940896a84948a04a5d56ef339c5f'
os.environ['LLM_BASE_URL'] = 'https://api.deepseek.com'
os.environ['LLM_MODEL'] = 'deepseek-chat'
os.environ['LLM_TEMPERATURE'] = '0.7'
os.environ['LLM_MAX_TOKENS'] = '2000'

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEDICAL_QA_DIR = os.path.join(SCRIPT_DIR, 'medical-qa-ragflow')

# 添加到Python路径
sys.path.insert(0, MEDICAL_QA_DIR)

# 导入并运行Web应用
if __name__ == '__main__':
    print("=" * 60)
    print("🤖 DeepSeek智能问答系统")
    print("=" * 60)
    print()
    print("📋 系统配置:")
    print(f"  • API Key: {os.environ['LLM_API_KEY'][:20]}...")
    print(f"  • Base URL: {os.environ['LLM_BASE_URL']}")
    print(f"  • Model: {os.environ['LLM_MODEL']}")
    print(f"  • Temperature: {os.environ['LLM_TEMPERATURE']}")
    print(f"  • Max Tokens: {os.environ['LLM_MAX_TOKENS']}")
    print()
    print("🌐 Web服务信息:")
    print("  • 服务地址: http://localhost:5001")
    print("  • 问答界面: http://localhost:5001/")
    print("  • API状态: http://localhost:5001/api/status")
    print("  • 健康检查: http://localhost:5001/health")
    print()
    print("⚠️  按 Ctrl+C 停止服务")
    print("=" * 60)
    print()
    
    # 导入并运行
    from deepseek_qa_web import app
    app.run(debug=True, host='0.0.0.0', port=5001)
