@echo off
echo 正在安装RAGFlow医学问答系统依赖...

REM 检查Python版本
python --version
if %errorlevel% neq 0 (
    echo 错误：未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo.
echo 升级pip到最新版本...
python -m pip install --upgrade pip

echo.
echo 安装核心依赖...
pip install torch>=1.9.0
pip install transformers>=4.20.0
pip install sentence-transformers>=2.2.0
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install scikit-learn>=1.0.0

echo.
echo 安装Web框架依赖...
pip install fastapi>=0.68.0
pip install uvicorn>=0.15.0

echo.
echo 安装向量数据库依赖...
pip install pymilvus>=2.0.0
pip install chromadb>=0.3.0

echo.
echo 安装文档处理依赖...
pip install pdf2image>=1.16.0
pip install pytesseract>=0.3.8
pip install pillow>=8.3.0

echo.
echo 安装自然语言处理依赖...
pip install spacy>=3.4.0
pip install jieba>=0.42.1

echo.
echo 安装API和工具依赖...
pip install openai>=0.27.0
pip install requests>=2.26.0
pip install python-multipart>=0.0.5
pip install aiofiles>=0.7.0
pip install python-dotenv>=0.19.0

echo.
echo 安装完成！现在可以下载预训练模型...
pause