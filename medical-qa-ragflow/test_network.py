#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的网络连接测试
"""

import socket
import sys

def test_connection(host, port=443, timeout=5):
    """测试网络连接"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"✅ 成功连接到 {host}:{port}")
            return True
        else:
            print(f"❌ 无法连接到 {host}:{port}")
            return False
    except Exception as e:
        print(f"❌ 连接测试失败: {e}")
        return False

print("=" * 60)
print("网络连接测试")
print("=" * 60)
print()

# 测试DNS解析
print("1. 测试DNS解析...")
try:
    import socket
    ip = socket.gethostbyname('api.deepseek.com')
    print(f"✅ DNS解析成功: api.deepseek.com -> {ip}")
except Exception as e:
    print(f"❌ DNS解析失败: {e}")
    sys.exit(1)

print()

# 测试HTTPS连接
print("2. 测试HTTPS连接 (443端口)...")
if test_connection('api.deepseek.com', 443):
    print("   HTTPS端口连接正常")
else:
    print("   HTTPS端口连接失败")

print()

# 测试HTTP连接
print("3. 测试HTTP连接 (80端口)...")
if test_connection('api.deepseek.com', 80):
    print("   HTTP端口连接正常")
else:
    print("   HTTP端口连接失败")

print()
print("=" * 60)
print("测试完成")
print("=" * 60)
