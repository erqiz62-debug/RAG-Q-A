@echo off
REM 便捷启动脚本 - 测试本地检索
echo ========================================
echo 测试本地检索功能
echo ========================================
echo.

cd /d "%~dp0medical-qa-ragflow"

if not exist "test_local_search.py" (
    echo 错误: 找不到 test_local_search.py 文件
    echo 请确保在正确的目录中运行此脚本
    pause
    exit /b 1
)

echo 当前目录: %CD%
echo.
echo 开始测试本地检索...
echo.

D:\Study\Python3.10\python.exe test_local_search.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo 测试完成！
    echo ========================================
) else (
    echo.
    echo ========================================
    echo 测试失败，错误代码: %ERRORLEVEL%
    echo ========================================
)

echo.
pause
