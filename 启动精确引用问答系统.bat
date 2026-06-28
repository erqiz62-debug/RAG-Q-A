@echo off
REM 便捷启动脚本 - 启动精确引用问答系统
echo ========================================
echo 启动精确引用医疗问答系统
echo ========================================
echo.

cd /d "%~dp0medical-qa-ragflow"

if not exist "precise_citation_medical_qa_system_local.py" (
    echo 错误: 找不到 precise_citation_medical_qa_system_local.py 文件
    echo 请确保在正确的目录中运行此脚本
    pause
    exit /b 1
)

echo 当前目录: %CD%
echo.
echo 启动精确引用问答系统...
echo 系统将运行在: http://localhost:8082
echo 按 Ctrl+C 停止服务
echo.

D:\Study\Python3.10\python.exe precise_citation_medical_qa_system_local.py

pause
