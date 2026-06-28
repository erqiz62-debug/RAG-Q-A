@echo off
echo 正在为Python 3.7安装RAGFlow医学问答系统依赖...

REM 检查Python版本
python --version

echo.
echo 升级pip到与Python 3.7兼容的版本...
python -m pip install --upgrade pip==23.3.1

echo.
echo 安装核心依赖（兼容Python 3.7）...
pip install torch==1.8.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.20.0
pip install sentence-transformers==2.2.0
pip install numpy==1.21.0
pip install pandas==1.3.0
pip install scikit-learn==1.0.0

echo.
echo 安装Web框架依赖...
pip install fastapi==0.68.0
pip install uvicorn==0.15.0

echo.
echo 安装向量数据库依赖...
pip install chromadb==0.3.0
pip install pymongo==3.12.0

echo.
echo 安装文档处理依赖...
pip install pdf2image==1.16.0
pip install pillow==8.3.0

echo.
echo 安装自然语言处理依赖...
pip install jieba==0.42.1
pip install opencc-python-reimplemented==0.1.6

echo.
echo 安装API和工具依赖...
pip install openai==0.27.0
pip install requests==2.26.0
pip install python-dotenv==0.19.0

echo.
echo 安装完成！请注意某些功能可能受限，建议升级到Python 3.8+以获得完整功能...
pause