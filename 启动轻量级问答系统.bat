@echo off
REM 便捷启动脚本 - 启动轻量级问答系统
echo ========================================
echo 启动轻量级医疗问答系统
echo ========================================
echo.

cd /d "%~dp0medical-qa-ragflow"

if not exist "lightweight_medical_qa_local.py" (
    echo 错误: 找不到 lightweight_medical_qa_local.py 文件
    echo 请确保在正确的目录中运行此脚本
    pause
    exit /b 1
)

echo 当前目录: %CD%
echo.
echo 启动轻量级问答系统...
echo 系统将运行在: http://localhost:5001
echo 按 Ctrl+C 停止服务
echo.

D:\Study\Python3.10\python.exe lightweight_medical_qa_local.py

pause
