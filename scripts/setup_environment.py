#!/usr/bin/env python3
"""
医学教材问答系统环境部署脚本
基于RAGFlow与DeepSeek的医学PDF智能问答系统
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGFlowDeployment:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.python_version = "3.8"
        
    def check_python_version(self):
        """检查Python版本"""
        current_version = sys.version_info
        logger.info(f"当前Python版本: {current_version.major}.{current_version.minor}.{current_version.micro}")
        
        if current_version < (3, 8):
            logger.error(f"需要Python {self.python_version}+，当前版本为{current_version.major}.{current_version.minor}")
            return False
        return True
    
    def install_dependencies(self):
        """安装必要的Python依赖"""
        dependencies = [
            "torch>=1.9.0",
            "transformers>=4.20.0",
            "sentence-transformers>=2.2.0",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "pymilvus>=2.0.0",
            "chromadb>=0.3.0",
            "pdf2image>=1.16.0",
            "pytesseract>=0.3.8",
            "pillow>=8.3.0",
            "spacy>=3.4.0",
            "jieba>=0.42.1",
            "openai>=0.27.0",
            "requests>=2.26.0",
            "python-multipart>=0.0.5",
            "aiofiles>=0.7.0",
            "python-dotenv>=0.19.0"
        ]
        
        for dep in dependencies:
            try:
                logger.info(f"安装依赖: {dep}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            except subprocess.CalledProcessError as e:
                logger.error(f"安装依赖失败: {dep}, 错误: {e}")
                return False
        return True
    
    def setup_vector_database(self):
        """设置向量数据库"""
        logger.info("设置向量数据库...")
        
        # 创建数据库配置
        db_config = {
            "vector_db_type": "chroma",  # 或 "milvus"
            "chroma": {
                "persist_directory": str(self.project_root / "data" / "chroma_db"),
                "collection_name": "medical_textbooks"
            },
            "milvus": {
                "host": "localhost",
                "port": 19530,
                "collection_name": "medical_textbooks"
            }
        }
        
        config_path = self.project_root / "configs" / "vector_db_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(db_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"数据库配置已保存到: {config_path}")
        return True
    
    def download_models(self):
        """下载预训练模型"""
        logger.info("下载预训练模型...")
        
        models_to_download = [
            "BAAI/bge-large-zh",  # 中文文本嵌入模型
            "BAAI/bge-reranker-large",  # 重排序模型
            "zh"  # 中文spaCy模型
        ]
        
        for model in models_to_download:
            try:
                if model == "zh":
                    logger.info(f"下载spaCy中文模型: {model}")
                    subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
                else:
                    logger.info(f"下载HuggingFace模型: {model}")
                    from sentence_transformers import SentenceTransformer
                    if "reranker" in model:
                        # 延迟加载reranker模型
                        continue
                    else:
                        SentenceTransformer(model)
            except Exception as e:
                logger.error(f"下载模型失败: {model}, 错误: {e}")
        
        return True
    
    def create_environment_file(self):
        """创建环境变量配置文件"""
        env_content = """# RAGFlow医学问答系统环境配置

# DeepSeek API配置
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

# 向量数据库配置
VECTOR_DB_TYPE=chroma
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
MILVUS_HOST=localhost
MILVUS_PORT=19530

# 模型配置
EMBEDDING_MODEL=BAAI/bge-large-zh
RERANKER_MODEL=BAAI/bge-reranker-large

# 系统配置
MAX_TOKENS=2048
TEMPERATURE=0.1
TOP_K=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=./logs/system.log
"""
        
        env_path = self.project_root / ".env"
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        logger.info(f"环境配置文件已创建: {env_path}")
        return True
    
    def run(self):
        """运行部署流程"""
        logger.info("开始部署RAGFlow医学问答系统...")
        
        # 检查Python版本
        if not self.check_python_version():
            return False
        
        # 创建日志目录
        (self.project_root / "logs").mkdir(exist_ok=True)
        
        # 安装依赖
        if not self.install_dependencies():
            return False
        
        # 设置向量数据库
        if not self.setup_vector_database():
            return False
        
        # 下载模型
        if not self.download_models():
            return False
        
        # 创建环境文件
        if not self.create_environment_file():
            return False
        
        logger.info("RAGFlow医学问答系统部署完成!")
        return True

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    deployment = RAGFlowDeployment(project_root)
    success = deployment.run()
    
    if success:
        logger.info("部署成功! 请查看配置文件并设置您的API密钥。")
    else:
        logger.error("部署失败，请检查错误日志。")
        sys.exit(1)