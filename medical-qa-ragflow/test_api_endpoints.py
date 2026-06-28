#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek API简化测试
"""

import requests
import json

API_KEY = 'sk-0553940896a84948a04a5d56ef339c5f'
BASE_URL = 'https://api.deepseek.com'

print("=" * 60)
print("DeepSeek API简化测试")
print("=" * 60)
print()

# 测试不同的端点
endpoints = [
    '/v1/chat/completions',
    '/chat/completions',
    '/v1/completions',
    '/completions'
]

for endpoint in endpoints:
    url = f'{BASE_URL}{endpoint}'
    print(f"测试端点: {url}")
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    payload = {
        'model': 'deepseek-chat',
        'messages': [
            {
                "role": "user",
                "content": "你好"
            }
        ],
        'max_tokens': 50
    }
    
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=10
        )
        
        print(f"  状态码: {response.status_code}")
        
        if response.status_code == 200:
            print(f"  ✅ 成功！")
            result = response.json()
            print(f"  响应: {json.dumps(result, ensure_ascii=False, indent=2)[:200]}...")
            print()
            print("=" * 60)
            print("找到正确的API端点！")
            print("=" * 60)
            break
        elif response.status_code == 401:
            print(f"  ❌ 认证失败 - API Key可能错误")
        elif response.status_code == 404:
            print(f"  ❌ 端点不存在")
        elif response.status_code == 429:
            print(f"  ❌ 请求过多 - 配额限制")
        else:
            print(f"  ❌ 失败: {response.text[:100]}")
            
    except requests.exceptions.Timeout:
        print(f"  ❌ 请求超时")
    except requests.exceptions.ConnectionError as e:
        print(f"  ❌ 连接错误: {e}")
    except Exception as e:
        print(f"  ❌ 错误: {e}")
    
    print()

print("=" * 60)
print("测试完成")
print("=" * 60)
