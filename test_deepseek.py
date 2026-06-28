#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试DeepSeek API连接
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

# 导入并运行测试脚本
if __name__ == '__main__':
    import test_deepseek_api
