# 医学教材问答系统 - 本地化版本

## 项目概述

基于RagFlow导出数据构建的本地医学教材问答系统，完全脱离RagFlow依赖，使用本地向量数据库进行检索。

## 核心特性

- ✅ **完全本地化**: 无需RagFlow容器，所有数据本地存储
- ✅ **混合检索**: 向量检索（70%）+ 关键词检索（30%）
- ✅ **精确溯源**: 支持文档名、页码、章节级别的溯源
- ✅ **高性能**: 使用ChromaDB向量数据库，支持大规模数据
- ✅ **易部署**: 单机部署，无需复杂配置
- ✅ **Python 3.7+兼容**: 支持较旧Python版本

## 项目结构

```
medical-qa-ragflow/
├── scripts/                          # 核心脚本
│   ├── local_data_loader.py           # 本地数据加载器
│   ├── local_vector_database.py       # 本地向量数据库
│   ├── precise_citation_medical_qa_system.py  # 精确溯源问答系统
│   └── ...
├── configs/                          # 配置文件
│   ├── local_config.json             # 本地配置（新增）
│   ├── ragflow_config.json          # RagFlow配置（已废弃）
│   ├── retrieval_config.json        # 检索策略
│   └── medical_chunking_config.json # 分块策略
├── data/                             # 数据目录
│   ├── chroma_db/                # ChromaDB向量数据库
│   ├── local_knowledge_base.json   # 本地知识库
│   └── test_report.json          # 测试报告
├── templates/                        # HTML模板
├── logs/                             # 日志目录
├── local_knowledge_base_builder.py  # 本地知识库构建器（新增）
├── lightweight_medical_qa_local.py  # 轻量级问答系统（新增）
├── test_local_retrieval.py        # 本地检索测试（新增）
├── app.py                          # Flask主应用（端口5001）
├── web_server.py                  # Web交互服务器（端口8082）
└── requirements_local.txt         # 本地版本依赖（新增）
```

## 快速开始

### 前置条件

1. **Python环境**: Python 3.7或更高版本
2. **数据准备**: RagFlow导出的分块数据（87个文件夹）
3. **依赖安装**: 安装必要的Python库

### 安装依赖

```bash
pip install -r requirements_local.txt
```

### 步骤1: 构建本地知识库

```bash
# 使用默认配置构建
python local_knowledge_base_builder.py

# 指定数据根目录
python local_knowledge_base_builder.py --data-root "D:\Study\py_program\导出分块\数据"

# 重新生成所有embeddings（需要安装sentence-transformers）
python local_knowledge_base_builder.py --regenerate-embeddings

# 增量更新知识库
python local_knowledge_base_builder.py --update
```

**构建过程**:
1. 📂 加载数据: 扫描87个文件夹，读取所有chunks.json和metadata.json
2. 🔢 处理向量: 使用现有embeddings或重新生成
3. 💾 构建向量数据库: 创建ChromaDB集合和倒排索引
4. 💾 导出知识库: 保存为local_knowledge_base.json

**输出**:
- `./data/chroma_db/`: ChromaDB向量数据库
- `./data/local_knowledge_base.json`: 本地知识库JSON文件
- `./data/build_report.json`: 构建报告

### 步骤2: 测试检索功能

```bash
# 运行测试脚本
python test_local_retrieval.py
```

**测试内容**:
1. 📂 数据加载测试
2. 💾 向量数据库测试
3. 🔍 向量检索测试
4. 🔑 关键词检索测试
5. 🔀 混合检索测试

### 步骤3: 启动Web服务

#### 方式一: Flask主应用（端口5001）

```bash
python app.py
```

访问地址: http://localhost:5001

**功能页面**:
- http://localhost:5001/ - 智能问答界面
- http://localhost:5001/dictionary - 医学词典界面
- http://localhost:5001/assessment - 智能自测界面
- http://localhost:5001/progress - 学习进度界面
- http://localhost:5001/comparison - 对比分析界面

#### 方式二: Web交互服务器（端口8082）

```bash
python web_server.py
```

访问地址: http://localhost:8082

**功能页面**:
- http://localhost:8082/ - 主界面
- http://localhost:8082/qa - 问答交互界面
- http://localhost:8082/demo - 演示界面
- http://localhost:8082/health - 健康检查

#### 方式三: 命令行问答

```bash
python lightweight_medical_qa_local.py
```

## 配置说明

### 本地配置文件

配置文件: `configs/local_config.json`

```json
{
  "local_system": {
    "enabled": true,
    "data_root": "D:\\Study\\py_program\\导出分块\\数据",
    "knowledge_base_file": "./data/local_knowledge_base.json",
    "vector_db_directory": "./data/chroma_db",
    "collection_name": "medical_knowledge"
  },
  "ragflow": {
    "enabled": false,
    "note": "RagFlow已禁用，系统完全本地化"
  },
  "retrieval": {
    "top_k": 5,
    "similarity_threshold": 0.3,
    "vector_weight": 0.7,
    "keyword_weight": 0.3,
    "method": "hybrid"
  },
  "web_server": {
    "app_port": 5001,
    "web_server_port": 8082,
    "host": "0.0.0.0",
    "debug": true
  }
}
```

### 环境变量

虽然系统主要使用配置文件，但仍支持部分环境变量：

```bash
# 数据根目录
export DATA_ROOT="D:\Study\py_program\导出分块\数据"

# 向量数据库目录
export VECTOR_DB_DIR="./data/chroma_db"

# Flask端口
export FLASK_APP_PORT=5001
export FLASK_WEB_SERVER_PORT=8082
```

## 核心功能

### 1. 本地数据加载器

**文件**: `scripts/local_data_loader.py`

**功能**:
- 遍历87个文件夹，加载所有chunks.json和metadata.json
- 提取chunk内容、元数据、文档信息
- 构建统计信息
- 导出为JSON文件

**使用示例**:
```python
from scripts.local_data_loader import LocalDataLoader

loader = LocalDataLoader("D:\\Study\\py_program\\导出分块\\数据")
data = loader.load_all_data()

print(f"文档数量: {data['total_documents']}")
print(f"Chunks数量: {data['total_chunks']}")
```

### 2. 本地向量数据库

**文件**: `scripts/local_vector_database.py`

**功能**:
- 使用ChromaDB存储向量（支持内存备用）
- 构建倒排索引用于关键词检索
- 支持向量检索、关键词检索、混合检索
- 自动降级到内存存储（如果ChromaDB不可用）

**使用示例**:
```python
from scripts.local_vector_database import LocalVectorDatabase

vector_db = LocalVectorDatabase("./data/chroma_db")

# 添加chunks
vector_db.add_chunks(chunks)

# 混合检索
results = vector_db.hybrid_search(
    query_embedding=query_embedding,
    query="高血压的诊断",
    top_k=5,
    vector_weight=0.7,
    keyword_weight=0.3
)
```

### 3. 本地知识库构建器

**文件**: `local_knowledge_base_builder.py`

**功能**:
- 一键构建本地知识库
- 支持增量更新
- 可选重新生成embeddings
- 生成构建报告

**命令行参数**:
```bash
python local_knowledge_base_builder.py --help
```

### 4. 轻量级问答系统

**文件**: `lightweight_medical_qa_local.py`

**功能**:
- 完全本地化的问答系统
- 混合检索策略
- 精确溯源（文档名、页码、章节）
- 命令行交互界面

**使用示例**:
```python
from lightweight_medical_qa_local import LightweightMedicalQALocal

qa_system = LightweightMedicalQALocal()
result = qa_system.answer_question("高血压的诊断标准")

print(f"答案: {result['答案']}")
print(f"置信度: {result['置信度']}")
print(f"来源: {result['来源']}")
print(f"页码: {result['页码']}")
```

## API接口

### Flask主应用（端口5001）

#### 智能问答
```http
POST /api/ask_question
Content-Type: application/json

{
  "question": "高血压的诊断标准是什么？"
}

Response:
{
  "answer": "根据《中国高血压防治指南》...",
  "confidence": 0.85,
  "timestamp": "2026-02-15T12:00:00Z"
}
```

#### 医学概念搜索
```http
POST /api/search_concepts
Content-Type: application/json

{
  "query": "高血压"
}

Response:
{
  "concepts": [
    {
      "name": "高血压",
      "definition": "以体循环动脉压升高为主要表现...",
      "symptoms": ["头痛", "头晕", "心悸"]
    }
  ]
}
```

### Web交互服务器（端口8082）

#### 问答接口
```http
POST /api/ask
Content-Type: application/json

{
  "question": "冠心病的治疗方法"
}

Response:
{
  "answer": "冠心病的治疗方法包括...",
  "confidence": 0.82,
  "sources": [
    {
      "document_title": "心脏病学.pdf",
      "page_number": 15,
      "snippet": "..."
    }
  ],
  "processing_time": 1.2
}
```

#### 系统状态
```http
GET /api/status

Response:
{
  "initialized": true,
  "total_questions": 100,
  "avg_confidence": 0.78,
  "knowledge_blocks": 870,
  "medical_terms": 5000
}
```

## 性能优化

### 内存优化

- 使用ChromaDB的持久化存储，减少内存占用
- 支持分批处理大规模数据
- 自动降级到内存存储（如果ChromaDB不可用）

### 检索优化

- 混合检索策略（向量+关键词）
- 倒排索引加速关键词检索
- 可配置的权重和阈值

### 数据处理优化

- 支持增量更新，无需重新构建整个知识库
- 批量生成embeddings
- 并行处理（可选）

## 故障排除

### 常见问题

#### 1. ChromaDB安装失败

**问题**: `ImportError: No module named 'chromadb'`

**解决**:
```bash
pip install chromadb
```

#### 2. 端口被占用

**问题**: `Address already in use`

**解决**:
- 修改 `configs/local_config.json` 中的端口配置
- 或者杀掉占用端口的进程

#### 3. 数据加载失败

**问题**: 找不到数据文件

**解决**:
- 检查数据根目录是否正确
- 确认chunks.json和metadata.json文件存在
- 检查文件权限

#### 4. 向量检索无结果

**问题**: 检索返回空结果

**解决**:
- 检查chunks是否有embedding字段
- 考虑重新生成embeddings: `--regenerate-embeddings`
- 调整相似度阈值

### 日志查看

```bash
# 查看系统日志
tail -f logs/local_system.log

# 查看Flask日志
tail -f logs/flask_app.log
```

## 迁移指南

### 从RagFlow迁移到本地版本

1. **导出数据**: 从RagFlow导出所有分块数据
2. **构建本地知识库**: 运行 `local_knowledge_base_builder.py`
3. **测试检索**: 运行 `test_local_retrieval.py`
4. **启动服务**: 运行 `app.py` 或 `web_server.py`

### 增量更新知识库

当有新的RagFlow导出数据时：

```bash
# 增量更新
python local_knowledge_base_builder.py --update --data-root "新数据路径"
```

## 技术架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    用户界面层                              │
├─────────────────────────────────────────────────────────────┤
│  Flask Web应用 (5001)  │  Web服务器 (8082)  │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                    业务逻辑层                              │
├─────────────────────────────────────────────────────────────┤
│  轻量级问答系统  │  精确溯源问答系统    │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                    数据存储层                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ ChromaDB    │  │ 倒排索引    │  │ JSON知识库  │ │
│  │ 向量数据库   │  │ 关键词检索  │  │ 元数据存储  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 数据流程图

```
RagFlow导出数据
       ↓
本地数据加载器
       ↓
本地向量数据库
       ↓
问答检索系统
       ↓
用户界面
```

## 贡献指南

### 代码规范

- 遵循PEP 8代码风格
- 添加类型提示
- 编写文档字符串
- 添加单元测试

### 提交流程

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用MIT许可证。

## 联系方式

- 项目主页: [GitHub仓库地址]
- 问题反馈: [Issues页面]
- 邮件: [联系邮箱]

---

**最后更新**: 2026-02-15  
**版本**: 2.0 (本地化版本)
