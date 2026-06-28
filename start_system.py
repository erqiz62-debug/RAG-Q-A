#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek智能问答系统 - Python启动脚本
"""

import os
import sys
import subprocess
import time

# 设置环境变量
os.environ['LLM_API_KEY'] = 'sk-0553940896a84948a04a5d56ef339c5f'
os.environ['LLM_BASE_URL'] = 'https://api.deepseek.com'
os.environ['LLM_MODEL'] = 'deepseek-chat'
os.environ['LLM_TEMPERATURE'] = '0.7'
os.environ['LLM_MAX_TOKENS'] = '2000'

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEDICAL_QA_DIR = os.path.join(SCRIPT_DIR, 'medical-qa-ragflow')

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")

def test_api_connection():
    """测试API连接"""
    print_header("测试DeepSeek API连接")
    
    test_script = os.path.join(MEDICAL_QA_DIR, 'test_deepseek_api.py')
    
    if not os.path.exists(test_script):
        print(f"❌ 错误: 找不到测试脚本 {test_script}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, test_script],
            cwd=MEDICAL_QA_DIR,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print(result.stdout)
        if result.stderr:
            print("错误输出:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ 测试超时")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def start_deepseek_qa():
    """启动DeepSeek问答系统"""
    print_header("启动DeepSeek智能问答系统")
    
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
    print()
    
    web_script = os.path.join(MEDICAL_QA_DIR, 'deepseek_qa_web.py')
    
    if not os.path.exists(web_script):
        print(f"❌ 错误: 找不到Web脚本 {web_script}")
        return False
    
    try:
        subprocess.run(
            [sys.executable, web_script],
            cwd=MEDICAL_QA_DIR
        )
    except KeyboardInterrupt:
        print("\n\n👋 服务已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False
    
    return True

def start_local_qa():
    """启动本地问答系统"""
    print_header("启动本地轻量级问答系统")
    
    print("🌐 Web服务信息:")
    print("  • 服务地址: http://localhost:5001")
    print("  • 问答界面: http://localhost:5001/")
    print()
    
    print("⚠️  按 Ctrl+C 停止服务")
    print()
    
    web_script = os.path.join(MEDICAL_QA_DIR, 'lightweight_medical_qa_local.py')
    
    if not os.path.exists(web_script):
        print(f"❌ 错误: 找不到Web脚本 {web_script}")
        return False
    
    try:
        subprocess.run(
            [sys.executable, web_script],
            cwd=MEDICAL_QA_DIR
        )
    except KeyboardInterrupt:
        print("\n\n👋 服务已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False
    
    return True

def start_precise_qa():
    """启动精确引用问答系统"""
    print_header("启动精确引用问答系统")
    
    print("🌐 Web服务信息:")
    print("  • 服务地址: http://localhost:8082")
    print("  • 问答界面: http://localhost:8082/")
    print()
    
    print("⚠️  按 Ctrl+C 停止服务")
    print()
    
    web_script = os.path.join(MEDICAL_QA_DIR, 'precise_citation_medical_qa_system_local.py')
    
    if not os.path.exists(web_script):
        print(f"❌ 错误: 找不到Web脚本 {web_script}")
        return False
    
    try:
        subprocess.run(
            [sys.executable, web_script],
            cwd=MEDICAL_QA_DIR
        )
    except KeyboardInterrupt:
        print("\n\n👋 服务已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False
    
    return True

def build_knowledge_base():
    """构建知识库"""
    print_header("构建知识库")
    
    build_script = os.path.join(MEDICAL_QA_DIR, 'build_knowledge_base_batch.py')
    
    if not os.path.exists(build_script):
        print(f"❌ 错误: 找不到构建脚本 {build_script}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, build_script],
            cwd=MEDICAL_QA_DIR
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 构建失败: {e}")
        return False

def test_search():
    """测试本地检索"""
    print_header("测试本地检索功能")
    
    test_script = os.path.join(MEDICAL_QA_DIR, 'test_local_search.py')
    
    if not os.path.exists(test_script):
        print(f"❌ 错误: 找不到测试脚本 {test_script}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, test_script],
            cwd=MEDICAL_QA_DIR
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def show_system_info():
    """显示系统信息"""
    print_header("系统信息")
    
    print("📋 基本信息:")
    print(f"  • Python版本: {sys.version}")
    print(f"  • 工作目录: {SCRIPT_DIR}")
    print(f"  • 系统目录: {MEDICAL_QA_DIR}")
    print()
    
    print("📚 知识库信息:")
    print("  • 数据目录: D:\\Study\\py_program\\导出分块\\数据")
    print("  • 向量数据库: ./data/chroma_db")
    print("  • 文档数量: 86个")
    print("  • 唯一chunks: 2482个")
    print("  • 倒排索引关键词: 42432个")
    print()
    
    print("🤖 DeepSeek配置:")
    print(f"  • API Key: {os.environ['LLM_API_KEY'][:20]}...")
    print(f"  • Base URL: {os.environ['LLM_BASE_URL']}")
    print(f"  • Model: {os.environ['LLM_MODEL']}")
    print(f"  • Temperature: {os.environ['LLM_TEMPERATURE']}")
    print(f"  • Max Tokens: {os.environ['LLM_MAX_TOKENS']}")
    print()
    
    print("🔧 系统功能:")
    print("  • 本地数据加载 (86个文档)")
    print("  • 向量数据库 (ChromaDB)")
    print("  • 混合检索 (向量 + 关键词)")
    print("  • DeepSeek智能问答系统 (端口 5001)")
    print("  • 本地轻量级问答系统 (端口 5001)")
    print("  • 精确引用问答系统 (端口 8082)")
    print()
    
    print("⚠️  注意事项:")
    print("  • DeepSeek API需要网络连接")
    print("  • 使用hash向量作为embedding备用方案")
    print("  • 关键词检索可能不如向量检索准确")

def main():
    """主菜单"""
    while True:
        print_header("医学问答系统 - 主菜单")
        
        print("  [1] 构建知识库")
        print("  [2] 测试本地检索功能")
        print("  [3] 启动DeepSeek智能问答系统 (端口 5001)")
        print("  [4] 启动本地轻量级问答系统 (端口 5001)")
        print("  [5] 启动精确引用问答系统 (端口 8082)")
        print("  [6] 测试DeepSeek API连接")
        print("  [7] 检查ChromaDB状态")
        print("  [8] 检查数据结构")
        print("  [9] 查看系统信息")
        print("  [0] 退出")
        print()
        
        choice = input("请选择操作 (0-9): ").strip()
        
        if choice == '1':
            build_knowledge_base()
        elif choice == '2':
            test_search()
        elif choice == '3':
            start_deepseek_qa()
        elif choice == '4':
            start_local_qa()
        elif choice == '5':
            start_precise_qa()
        elif choice == '6':
            test_api_connection()
        elif choice == '7':
            check_script = os.path.join(MEDICAL_QA_DIR, 'check_chromadb.py')
            if os.path.exists(check_script):
                subprocess.run([sys.executable, check_script], cwd=MEDICAL_QA_DIR)
            else:
                print("❌ 找不到检查脚本")
        elif choice == '8':
            check_script = os.path.join(MEDICAL_QA_DIR, 'check_chunks_structure.py')
            if os.path.exists(check_script):
                subprocess.run([sys.executable, check_script], cwd=MEDICAL_QA_DIR)
            else:
                print("❌ 找不到检查脚本")
        elif choice == '9':
            show_system_info()
        elif choice == '0':
            print("\n👋 感谢使用医学问答系统！\n")
            break
        else:
            print("\n❌ 无效的选择，请重新输入\n")
        
        input("\n按回车键继续...")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
