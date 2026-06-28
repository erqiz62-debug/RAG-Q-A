# 🚀 Python启动脚本

## 快速开始

### 方式一：使用主菜单（推荐）

```bash
python start_system.py
```

### 方式二：直接启动特定系统

```bash
# 启动DeepSeek智能问答系统
python start_deepseek_qa.py

# 启动本地轻量级问答系统
python start_local_qa.py

# 启动精确引用问答系统
python start_precise_qa.py
```

## 📋 可用脚本

| 脚本 | 功能 | 端口 |
|------|------|------|
| [start_system.py](file:///D:/Study/py_program/基于ragflow版/start_system.py) | 主菜单（所有功能） | - |
| [start_deepseek_qa.py](file:///D:/Study/py_program/基于ragflow版/start_deepseek_qa.py) | DeepSeek智能问答 | 5001 |
| [start_local_qa.py](file:///D:/Study/py_program/基于ragflow版/start_local_qa.py) | 本地轻量级问答 | 5001 |
| [start_precise_qa.py](file:///D:/Study/py_program/基于ragflow版/start_precise_qa.py) | 精确引用问答 | 8082 |
| [test_deepseek.py](file:///D:/Study/py_program/基于ragflow版/test_deepseek.py) | 测试DeepSeek API | - |
| [build_knowledge_base.py](file:///D:/Study/py_program/基于ragflow版/build_knowledge_base.py) | 构建知识库 | - |

## 🌐 访问地址

### DeepSeek智能问答系统
- 主页: http://localhost:5001
- API状态: http://localhost:5001/api/status
- 健康检查: http://localhost:5001/health

### 本地轻量级问答系统
- 主页: http://localhost:5001

### 精确引用问答系统
- 主页: http://localhost:8082

## ⚙️ 系统配置

### DeepSeek API
```
API Key: sk-0553940896a84948a04a5d56ef339c5f
Base URL: https://api.deepseek.com
Model: deepseek-chat
Temperature: 0.7
Max Tokens: 2000
```

### 知识库
```
数据目录: D:\Study\py_program\导出分块\数据
向量数据库: ./data/chroma_db
文档数量: 86个
唯一chunks: 2482个
倒排索引关键词: 42432个
```

## 📖 详细文档

查看 [Python启动脚本使用指南.md](file:///D:/Study/py_program/基于ragflow版/Python启动脚本使用指南.md) 获取详细说明。

## ⚠️ 注意事项

- DeepSeek问答系统和本地轻量级问答系统都使用端口5001，不能同时运行
- 精确引用问答系统使用端口8082
- DeepSeek API需要网络连接
- 确保Python版本为3.10

## 🎯 首次使用

1. 测试API连接
   ```bash
   python test_deepseek.py
   ```

2. 构建知识库（如果还没构建）
   ```bash
   python build_knowledge_base.py
   ```

3. 启动系统
   ```bash
   python start_deepseek_qa.py
   ```

4. 访问Web界面
   - 打开浏览器访问：http://localhost:5001

---

**Python版本**: 3.10  
**最后更新**: 2026-02-15
