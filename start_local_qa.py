#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接启动本地轻量级问答系统
"""

import os
import sys

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEDICAL_QA_DIR = os.path.join(SCRIPT_DIR, 'medical-qa-ragflow')

# 添加到Python路径
sys.path.insert(0, MEDICAL_QA_DIR)

# 导入并运行Web应用
if __name__ == '__main__':
    print("=" * 60)
    print("🏥 本地轻量级问答系统")
    print("=" * 60)
    print()
    print("🌐 Web服务信息:")
    print("  • 服务地址: http://localhost:5001")
    print("  • 问答界面: http://localhost:5001/")
    print()
    print("⚠️  按 Ctrl+C 停止服务")
    print("=" * 60)
    print()
    
    # 导入并运行
    from lightweight_medical_qa_local import app
    app.run(debug=True, host='0.0.0.0', port=5001)
