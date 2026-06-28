@echo off
chcp 65001 >nul
title 医疗问答系统 - 主菜单

:MENU
cls
echo ========================================
echo     医疗问答系统 - 主菜单
echo ========================================
echo.
echo  [1] 构建知识库
echo  [2] 测试本地检索功能
echo  [3] 启动轻量级问答系统 (端口 5001)
echo  [4] 启动精确引用问答系统 (端口 8082)
echo  [5] 启动DeepSeek智能问答系统 (端口 5001)
echo  [6] 测试DeepSeek API连接
echo  [7] 检查ChromaDB状态
echo  [8] 检查数据结构
echo  [9] 查看系统信息
echo  [0] 退出
echo.
echo ========================================
set /p choice="请选择操作 (0-9): "

if "%choice%"=="1" goto BUILD_KB
if "%choice%"=="2" goto TEST_SEARCH
if "%choice%"=="3" goto START_LIGHTWEIGHT
if "%choice%"=="4" goto START_PRECISE
if "%choice%"=="5" goto START_DEEPSEEK
if "%choice%"=="6" goto TEST_DEEPSEEK_API
if "%choice%"=="7" goto CHECK_CHROMADB
if "%choice%"=="8" goto CHECK_DATA
if "%choice%"=="9" goto SHOW_INFO
if "%choice%"=="0" goto EXIT
goto MENU

:BUILD_KB
cls
echo ========================================
echo 构建知识库
echo ========================================
echo.
cd /d "%~dp0medical-qa-ragflow"
D:\Study\Python3.10\python.exe build_knowledge_base_batch.py
echo.
pause
goto MENU

:TEST_SEARCH
cls
echo ========================================
echo 测试本地检索功能
echo ========================================
echo.
cd /d "%~dp0medical-qa-ragflow"
D:\Study\Python3.10\python.exe test_local_search.py
echo.
pause
goto MENU

:START_LIGHTWEIGHT
cls
echo ========================================
echo 启动轻量级问答系统
echo ========================================
echo.
echo 系统将运行在: http://localhost:5001
echo 按 Ctrl+C 停止服务
echo.
cd /d "%~dp0medical-qa-ragflow"
D:\Study\Python3.10\python.exe lightweight_medical_qa_local.py
pause
goto MENU

:START_PRECISE
cls
echo ========================================
echo 启动精确引用问答系统
echo ========================================
echo.
echo 系统将运行在: http://localhost:8082
echo 按 Ctrl+C 停止服务
echo.
cd /d "%~dp0medical-qa-ragflow"
D:\Study\Python3.10\python.exe precise_citation_medical_qa_system_local.py
pause
goto MENU

:START_DEEPSEEK
cls
echo ========================================
echo 启动DeepSeek智能问答系统
echo ========================================
echo.
echo 系统将运行在: http://localhost:5001
echo 按 Ctrl+C 停止服务
echo.
cd /d "%~dp0medical-qa-ragflow"
D:\Study\Python3.10\python.exe deepseek_qa_web.py
pause
goto MENU

:TEST_DEEPSEEK_API
cls
echo ========================================
echo 测试DeepSeek API连接
echo ========================================
echo.
cd /d "%~dp0medical-qa-ragflow"
D:\Study\Python3.10\python.exe test_deepseek_api.py
echo.
pause
goto MENU

:CHECK_CHROMADB
cls
echo ========================================
echo 检查ChromaDB状态
echo ========================================
echo.
cd /d "%~dp0medical-qa-ragflow"
D:\Study\Python3.10\python.exe check_chromadb.py
echo.
pause
goto MENU

:CHECK_DATA
cls
echo ========================================
echo 检查数据结构
echo ========================================
echo.
cd /d "%~dp0medical-qa-ragflow"
D:\Study\Python3.10\python.exe check_chunks_structure.py
echo.
pause
goto MENU

:SHOW_INFO
cls
echo ========================================
echo 系统信息
echo ========================================
echo.
echo Python版本: 3.10
echo 数据目录: D:\Study\py_program\导出分块\数据
echo 知识库目录: D:\Study\py_program\基于ragflow版\medical-qa-ragflow\data\chroma_db
echo.
echo 系统功能:
echo   - 本地数据加载 (86个文档)
echo   - 向量数据库 (ChromaDB)
echo   - 混合检索 (向量 + 关键词)
echo   - 轻量级问答系统 (端口 5001)
echo   - 精确引用问答系统 (端口 8082)
echo   - DeepSeek智能问答系统 (端口 5001)
echo.
echo DeepSeek配置:
echo   - API Key: sk-0553940896a84948a04a5d56ef339c5f
echo   - Base URL: https://api.deepseek.com
echo   - Model: deepseek-chat
echo   - Temperature: 0.7
echo   - Max Tokens: 2000
echo.
echo 已知问题:
echo   - 使用hash向量作为embedding备用方案
echo   - 关键词检索可能不如向量检索准确
echo   - DeepSeek API需要网络连接
echo.
pause
goto MENU

:EXIT
cls
echo 感谢使用医疗问答系统！
echo.
timeout /t 2 >nul
exit
