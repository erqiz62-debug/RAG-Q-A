@echo off
REM 便捷启动脚本 - 测试DeepSeek API
echo ========================================
echo 测试DeepSeek API连接
echo ========================================
echo.

cd /d "%~dp0medical-qa-ragflow"

if not exist "test_deepseek_api.py" (
    echo 错误: 找不到 test_deepseek_api.py 文件
    echo 请确保在正确的目录中运行此脚本
    pause
    exit /b 1
)

echo 当前目录: %CD%
echo.
echo 开始测试DeepSeek API...
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

D:\Study\Python3.10\python.exe test_deepseek_api.py

echo.
pause
