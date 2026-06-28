@echo off
REM 便捷启动脚本 - DeepSeek智能问答系统
echo ========================================
echo 启动DeepSeek智能问答系统
echo ========================================
echo.

cd /d "%~dp0medical-qa-ragflow"

if not exist "deepseek_qa_web.py" (
    echo 错误: 找不到 deepseek_qa_web.py 文件
    echo 请确保在正确的目录中运行此脚本
    pause
    exit /b 1
)

echo 当前目录: %CD%
echo.
echo 启动DeepSeek智能问答系统...
echo 系统将运行在: http://localhost:5001
echo 按 Ctrl+C 停止服务
echo.

REM 设置环境变量
set LLM_API_KEY=sk-0553940896a84948a04a5d56ef339c5f
set LLM_BASE_URL=https://api.deepseek.com
set LLM_MODEL=deepseek-chat
set LLM_TEMPERATURE=0.7
set LLM_MAX_TOKENS=2000

echo API配置:
echo   API Key: %LLM_API_KEY%
echo   Base URL: %LLM_BASE_URL%
echo   Model: %LLM_MODEL%
echo   Temperature: %LLM_TEMPERATURE%
echo   Max Tokens: %LLM_MAX_TOKENS%
echo.

D:\Study\Python3.10\python.exe deepseek_qa_web.py

pause
