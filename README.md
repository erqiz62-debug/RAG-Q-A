# 医学教材问答系统（基于RAGFlow与DeepSeek）

## 项目概述
基于RAGFlow开源框架构建的面向医学教材的专业智能问答系统，答案由DeepSeek大模型提供。

## 核心特性
- 深度解析医学PDF教材，保证知识单元语义完整性
- 精准溯源到教材名称和页码
- 混合检索（向量检索+关键词检索）
- 基于bge-reranker的重排序模型
- 专业的医学问答提示词模板

## 项目结构
```
medical-qa-ragflow/
├── docs/                 # 文档目录
├── configs/             # 配置文件
├── data/               # 数据目录
│   ├── pdfs/          # 医学PDF教材
│   └── test_sets/     # 测试集
├── models/             # 模型配置
├── scripts/            # 部署和配置脚本
├── knowledge_base/     # RAGFlow知识库配置
└── reports/            # 测试报告
```

## 执行阶段

### 第一阶段：知识库构建
- [ ] 部署RAGFlow环境
- [ ] 配置向量数据库（Milvus/Chroma）
- [ ] 创建医学PDF知识库
- [ ] 启用深度解析模式
- [ ] 配置语义分块策略
- [ ] 执行向量化入库

### 第二阶段：智能问答引擎
- [ ] 配置混合检索策略
- [ ] 设置重排序模型
- [ ] 集成DeepSeek API
- [ ] 设计提示词模板
- [ ] 验证溯源功能

### 第三阶段：系统评估与优化
- [ ] 功能与性能测试
- [ ] 医学准确性评估
- [ ] 参数迭代优化
- [ ] 生成配置文档

## 依赖环境
- RAGFlow
- Milvus 或 Chroma 向量数据库
- DeepSeek API
- Python 3.8+
- bge-large-zh 嵌入模型
- bge-reranker 重排序模型