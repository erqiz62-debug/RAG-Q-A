#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建知识库
"""

import os
import sys

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEDICAL_QA_DIR = os.path.join(SCRIPT_DIR, 'medical-qa-ragflow')

# 添加到Python路径
sys.path.insert(0, MEDICAL_QA_DIR)

# 导入并运行构建脚本
if __name__ == '__main__':
    print("=" * 60)
    print("📚 构建知识库")
    print("=" * 60)
    print()
    
    import build_knowledge_base_batch
