@echo off
REM 便捷启动脚本 - 构建知识库
echo ========================================
echo 构建本地知识库
echo ========================================
echo.

cd /d "%~dp0medical-qa-ragflow"

if not exist "build_knowledge_base_batch.py" (
    echo 错误: 找不到 build_knowledge_base_batch.py 文件
    echo 请确保在正确的目录中运行此脚本
    pause
    exit /b 1
)

echo 当前目录: %CD%
echo.
echo 开始构建知识库...
echo.

D:\Study\Python3.10\python.exe build_knowledge_base_batch.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo 知识库构建完成！
    echo ========================================
) else (
    echo.
    echo ========================================
    echo 知识库构建失败，错误代码: %ERRORLEVEL%
    echo ========================================
)

echo.
pause
