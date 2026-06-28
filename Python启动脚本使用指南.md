# 🚀 Python启动脚本使用指南

## 📋 概述

本项目提供了多个Python启动脚本，可以直接运行，无需使用.bat批处理文件。

## 🎯 可用的启动脚本

### 1. 主菜单脚本

**[start_system.py](file:///D:/Study/py_program/基于ragflow版/start_system.py)** - 主菜单启动脚本

提供交互式菜单，包含所有功能：

```bash
python start_system.py
```

菜单选项：
- [1] 构建知识库
- [2] 测试本地检索功能
- [3] 启动DeepSeek智能问答系统 (端口 5001)
- [4] 启动本地轻量级问答系统 (端口 5001)
- [5] 启动精确引用问答系统 (端口 8082)
- [6] 测试DeepSeek API连接
- [7] 检查ChromaDB状态
- [8] 检查数据结构
- [9] 查看系统信息
- [0] 退出

### 2. 单独启动脚本

#### DeepSeek智能问答系统

**[start_deepseek_qa.py](file:///D:/Study/py_program/基于ragflow版/start_deepseek_qa.py)** - 直接启动DeepSeek问答系统

```bash
python start_deepseek_qa.py
```

访问地址：http://localhost:5001

#### 本地轻量级问答系统

**[start_local_qa.py](file:///D:/Study/py_program/基于ragflow版/start_local_qa.py)** - 直接启动本地问答系统

```bash
python start_local_qa.py
```

访问地址：http://localhost:5001

#### 精确引用问答系统

**[start_precise_qa.py](file:///D:/Study/py_program/基于ragflow版/start_precise_qa.py)** - 直接启动精确引用系统

```bash
python start_precise_qa.py
```

访问地址：http://localhost:8082

### 3. 测试和工具脚本

#### 测试DeepSeek API

**[test_deepseek.py](file:///D:/Study/py_program/基于ragflow版/test_deepseek.py)** - 测试API连接

```bash
python test_deepseek.py
```

#### 构建知识库

**[build_knowledge_base.py](file:///D:/Study/py_program/基于ragflow版/build_knowledge_base.py)** - 构建知识库

```bash
python build_knowledge_base.py
```

## 📖 使用方法

### 方法一：使用主菜单（推荐）

1. 打开命令行/终端
2. 进入项目目录：
   ```bash
   cd D:\Study\py_program\基于ragflow版
   ```
3. 运行主菜单：
   ```bash
   python start_system.py
   ```
4. 根据提示选择操作

### 方法二：直接启动特定系统

1. 打开命令行/终端
2. 进入项目目录：
   ```bash
   cd D:\Study\py_program\基于ragflow版
   ```
3. 运行对应的启动脚本：
   ```bash
   # 启动DeepSeek问答系统
   python start_deepseek_qa.py
   
   # 或启动本地问答系统
   python start_local_qa.py
   
   # 或启动精确引用系统
   python start_precise_qa.py
   ```

### 方法三：在IDE中运行

1. 在IDE中打开对应的Python文件
2. 直接运行文件
3. 查看控制台输出获取访问地址

## 🔧 系统配置

### DeepSeek API配置

所有启动脚本已内置以下配置：

```python
LLM_API_KEY = 'sk-0553940896a84948a04a5d56ef339c5f'
LLM_BASE_URL = 'https://api.deepseek.com'
LLM_MODEL = 'deepseek-chat'
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 2000
```

### 知识库配置

- **数据目录**: `D:\Study\py_program\导出分块\数据`
- **向量数据库**: `./data/chroma_db`
- **集合名称**: `medical_knowledge`
- **文档数量**: 86个
- **唯一chunks**: 2482个
- **倒排索引关键词**: 42432个

## 🌐 Web界面访问

### DeepSeek智能问答系统

- **服务地址**: http://localhost:5001
- **问答界面**: http://localhost:5001/
- **API状态**: http://localhost:5001/api/status
- **健康检查**: http://localhost:5001/health

### 本地轻量级问答系统

- **服务地址**: http://localhost:5001
- **问答界面**: http://localhost:5001/

### 精确引用问答系统

- **服务地址**: http://localhost:8082
- **问答界面**: http://localhost:8082/

## 📊 快速开始

### 首次使用

1. **测试API连接**
   ```bash
   python test_deepseek.py
   ```

2. **构建知识库**（如果还没构建）
   ```bash
   python build_knowledge_base.py
   ```

3. **启动DeepSeek问答系统**
   ```bash
   python start_deepseek_qa.py
   ```

4. **访问Web界面**
   - 打开浏览器访问：http://localhost:5001

### 日常使用

直接运行：
```bash
python start_deepseek_qa.py
```

或使用主菜单：
```bash
python start_system.py
```

## ⚠️ 注意事项

### 端口冲突

- DeepSeek问答系统和本地轻量级问答系统都使用端口 **5001**
- 精确引用问答系统使用端口 **8082**
- 如果端口被占用，请先停止其他服务

### 网络要求

- DeepSeek API需要网络连接
- 确保能够访问 `api.deepseek.com`
- 检查防火墙设置

### Python版本

- 推荐使用 Python 3.10
- 确保已安装所有依赖包

## 🛠️ 故障排除

### 问题1：无法启动服务

**解决方案**：
- 检查端口是否被占用
- 确认Python版本正确
- 检查依赖包是否安装完整

### 问题2：API连接失败

**解决方案**：
- 运行测试脚本：`python test_deepseek.py`
- 检查网络连接
- 验证API Key是否正确
- 确认API地址是否正确

### 问题3：知识库检索失败

**解决方案**：
- 确认知识库已构建：`python build_knowledge_base.py`
- 检查数据目录是否存在
- 验证向量数据库文件

### 问题4：Web界面无法访问

**解决方案**：
- 确认服务已启动
- 检查端口是否正确
- 尝试使用 `127.0.0.1` 而不是 `localhost`
- 检查防火墙设置

## 📝 脚本说明

### start_system.py

主菜单脚本，提供交互式菜单界面。

**功能**：
- 统一入口，集成所有功能
- 自动配置环境变量
- 显示系统信息
- 提供友好的用户界面

### start_deepseek_qa.py

直接启动DeepSeek智能问答系统。

**功能**：
- 自动配置DeepSeek API参数
- 启动Flask Web服务
- 显示服务配置信息
- 提供访问地址

### start_local_qa.py

直接启动本地轻量级问答系统。

**功能**：
- 启动本地Flask Web服务
- 使用本地知识库
- 显示服务信息

### start_precise_qa.py

直接启动精确引用问答系统。

**功能**：
- 启动精确引用Flask Web服务
- 提供精确的引用信息
- 显示服务信息

### test_deepseek.py

测试DeepSeek API连接。

**功能**：
- 测试API连接
- 验证API Key
- 显示测试结果

### build_knowledge_base.py

构建知识库。

**功能**：
- 从导出数据加载chunks
- 构建向量数据库
- 构建倒排索引
- 显示构建进度

## 🎓 最佳实践

### 1. 使用主菜单

对于日常使用，推荐使用主菜单脚本：
```bash
python start_system.py
```

### 2. 直接启动特定系统

如果需要频繁使用某个系统，可以直接运行对应的启动脚本：
```bash
python start_deepseek_qa.py
```

### 3. 测试后再使用

在使用DeepSeek问答系统前，先测试API连接：
```bash
python test_deepseek.py
```

### 4. 定期更新知识库

定期构建和更新知识库：
```bash
python build_knowledge_base.py
```

## 📞 技术支持

如果遇到问题：

1. **查看文档**
   - [DeepSeek问答系统说明.md](file:///D:/Study/py_program/基于ragflow版/DeepSeek问答系统说明.md)
   - [DeepSeek系统完成报告.md](file:///D:/Study/py_program/基于ragflow版/DeepSeek系统完成报告.md)

2. **运行测试**
   - 测试API：`python test_deepseek.py`
   - 测试网络：`python medical-qa-ragflow/test_network.py`
   - 测试端点：`python medical-qa-ragflow/test_api_endpoints.py`

3. **检查日志**
   - 查看控制台输出
   - 检查错误信息
   - 查看系统状态

## 🎊 总结

现在您可以使用纯Python脚本来启动系统，无需.bat文件！

**主要优势**：
- ✅ 跨平台兼容
- ✅ 更好的错误处理
- ✅ 更清晰的输出
- ✅ 更容易调试
- ✅ 可以在IDE中直接运行

**快速开始**：
```bash
# 使用主菜单
python start_system.py

# 或直接启动DeepSeek问答系统
python start_deepseek_qa.py
```

---

**最后更新**: 2026-02-15  
**Python版本**: 3.10  
**状态**: ✅ 已完成
