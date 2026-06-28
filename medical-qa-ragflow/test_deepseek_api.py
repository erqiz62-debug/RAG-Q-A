#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试DeepSeek API连接
"""

import os
import sys
import requests
import json

# DeepSeek API配置
API_KEY = 'sk-0553940896a84948a04a5d56ef339c5f'
BASE_URL = 'https://api.deepseek.com'
MODEL = 'deepseek-chat'
TEMPERATURE = 0.7
MAX_TOKENS = 2000

# 设置环境变量（供其他模块使用）
os.environ['LLM_API_KEY'] = API_KEY
os.environ['LLM_BASE_URL'] = BASE_URL
os.environ['LLM_MODEL'] = MODEL
os.environ['LLM_TEMPERATURE'] = str(TEMPERATURE)
os.environ['LLM_MAX_TOKENS'] = str(MAX_TOKENS)

print("=" * 60)
print("DeepSeek API连接测试")
print("=" * 60)
print()

print("API配置:")
print(f"  API Key: {API_KEY[:20]}...")
print(f"  Base URL: {BASE_URL}")
print(f"  Model: {MODEL}")
print(f"  Temperature: {TEMPERATURE}")
print(f"  Max Tokens: {MAX_TOKENS}")
print()

# 测试问题
test_question = "什么是心脏病？"

print(f"测试问题: {test_question}")
print()

try:
    # 构建请求
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    payload = {
        'model': MODEL,
        'messages': [
            {
                "role": "system",
                "content": "你是一个专业的医学知识问答助手，基于医学教材和权威资料回答用户的问题。"
            },
            {
                "role": "user",
                "content": test_question
            }
        ],
        'temperature': TEMPERATURE,
        'max_tokens': MAX_TOKENS
    }
    
    print("发送API请求...")
    print(f"请求URL: {BASE_URL}/v1/chat/completions")
    print(f"请求超时: 30秒")
    
    response = None
    try:
        response = requests.post(
            f'{BASE_URL}/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=30,
            verify=False  # 跳过SSL验证（如果遇到SSL问题）
        )
    except requests.exceptions.SSLError as e:
        print(f"❌ SSL错误: {e}")
        print("提示: 尝试使用 http:// 而不是 https://")
        sys.exit(1)
    except requests.exceptions.Timeout as e:
        print(f"❌ 请求超时: {e}")
        print("提示: 检查网络连接或增加超时时间")
        sys.exit(1)
    except requests.exceptions.ConnectionError as e:
        print(f"❌ 连接错误: {e}")
        print("提示: 检查网络连接和API地址")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"响应状态码: {response.status_code}")
    print(f"响应头: {dict(response.headers)}")
    print()
    
    if response.status_code == 200:
        result = response.json()
        answer = result['choices'][0]['message']['content']
        
        print("✅ API调用成功！")
        print()
        print("回答:")
        print("-" * 60)
        print(answer)
        print("-" * 60)
        print()
        
        # 显示使用信息
        usage = result.get('usage', {})
        if usage:
            print("使用统计:")
            print(f"  提示词tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"  完成tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"  总tokens: {usage.get('total_tokens', 'N/A')}")
        
    else:
        print("❌ API调用失败！")
        print(f"错误信息: {response.text}")
        
except requests.exceptions.Timeout:
    print("❌ 请求超时！请检查网络连接。")
except requests.exceptions.ConnectionError:
    print("❌ 连接错误！请检查网络连接和API地址。")
except Exception as e:
    print(f"❌ 发生错误: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("测试完成")
print("=" * 60)
