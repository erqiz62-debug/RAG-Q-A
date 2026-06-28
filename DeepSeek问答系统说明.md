# DeepSeek智能医学问答系统

## 🎯 系统简介

DeepSeek智能医学问答系统是一个基于DeepSeek大模型的智能问答服务，结合本地知识库检索和AI大模型生成，提供专业、准确的医学问答服务。

### 主要特点

- ✅ **智能问答**: 使用DeepSeek大模型生成专业答案
- ✅ **知识库检索**: 模拟基于本地知识库的检索
- ✅ **Web界面**: 提供友好的Web界面
- ✅ **API接口**: 支持RESTful API调用
- ✅ **混合检索**: 结合向量和关键词检索
- ✅ **实时响应**: 快速响应用户问题

## 🔧 系统配置

### DeepSeek API配置

```bash
LLM_API_KEY=sk-0553940896a84948a04a5d56ef339c5f
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000
```

### 知识库配置

- **数据目录**: `D:\Study\py_program\导出分块\数据`
- **向量数据库**: `./data/chroma_db`
- **集合名称**: `medical_knowledge`
- **文档数量**: 86个
- **唯一chunks**: 2482个
- **倒排索引关键词**: 42432个

## 🚀 快速开始

### 方法一：使用主菜单（推荐）

1. 双击运行 `主菜单.bat`
2. 选择 `[6] 测试DeepSeek API连接` - 测试API是否正常
3. 选择 `[5] 启动DeepSeek智能问答系统` - 启动Web服务
4. 在浏览器中访问 `http://localhost:5001`

### 方法二：直接运行脚本

1. **测试API连接**
   - 双击 `测试DeepSeek API.bat`

2. **启动Web服务**
   - 双击 `启动DeepSeek问答系统.bat`

3. **访问Web界面**
   - 打开浏览器访问 `http://localhost:5001`

## 📁 文件说明

### 核心文件

- **[deepseek_medical_qa.py](file:///d:/Study/py_program/基于ragflow版/medical-qa-ragflow/deepseek_medical_qa.py)** - DeepSeek问答系统核心类
- **[deepseek_qa_web.py](file:///d:/Study/py_program/基于ragflow版/medical-qa-ragflow/deepseek_qa_web.py)** - Flask Web应用
- **[test_deepseek_api.py](file:///d:/Study/py_program/基于ragflow版/medical-qa-ragflow/test_deepseek_api.py)** - API测试脚本

### 便捷脚本

- **[启动DeepSeek问答系统.bat](file:///D:/Study/py_program/基于ragflow版/启动DeepSeek问答系统.bat)** - 启动Web服务
- **[测试DeepSeek API.bat](file:///D:/Study/py_program/基于ragflow版/测试DeepSeek API.bat)** - 测试API连接
- **[主菜单.bat](file:///D:/Study/py_program/基于ragflow版/主菜单.bat)** - 主菜单（包含所有功能）

## 🌐 Web界面功能

### 主要功能

1. **问答界面**
   - 输入医学问题
   - 选择是否使用知识库检索
   - 设置检索结果数量
   - 获取AI生成的专业答案

2. **答案展示**
   - 显示问题
   - 显示AI生成的答案
   - 显示置信度
   - 显示知识来源（如果使用知识库）
   - 显示回答时间

3. **系统状态**
   - 问答引擎信息
   - 模型配置
   - 知识库统计
   - 系统时间

### API接口

#### 1. 获取系统状态

```
GET /api/status
```

响应示例：
```json
{
  "success": true,
  "result": {
    "系统状态": "正常运行",
    "问答引擎": "DeepSeek API",
    "API配置": {
      "基础URL": "https://api.deepseek.com",
      "模型": "deepseek-chat",
      "温度": 0.7,
      "最大token数": 2000
    },
    "知识库条目": 2482,
    "倒排索引大小": 42432
  }
}
```

#### 2. 问答接口

```
POST /api/answer
Content-Type: application/json

{
  "question": "什么是心脏病？",
  "use_knowledge": true,
  "top_k": 3
}
```

响应示例：
```json
{
  "success": true,
  "result": {
    "问题": "什么是心脏病？",
    "答案": "心脏病是指心脏结构或功能异常的一类疾病...",
    "置信度": 0.95,
    "使用知识库": true,
    "检索到的知识数量": 3,
    "知识来源": [
      {
        "document": "心脏病学.pdf",
        "page": 1,
        "score": 0.85
      }
    ],
    "回答时间": "2026-02-15 14:30:00"
  }
}
```

#### 3. 批量问答接口

```
POST /api/batch_answer
Content-Type: application/json

{
  "questions": [
    "什么是心脏病？",
    "冠心病的症状有哪些？"
  ]
}
```

#### 4. 健康检查

```
GET /health
```

响应示例：
```json
{
  "status": "healthy",
  "timestamp": "2026-02-15 14:30:00"
}
```

## 🔍 工作原理

### 1. 知识库检索（模拟）

当用户选择使用知识库时，系统会：

1. 从本地向量数据库检索相关内容
2. 使用混合检索（向量+关键词）
3. 返回最相关的chunks
4. 将检索结果作为上下文提供给DeepSeek

### 2. DeepSeek问答

系统会：

1. 构建系统提示词（专业医学助手）
2. 将用户问题和检索到的知识组合
3. 调用DeepSeek API生成答案
4. 返回格式化的结果

### 3. 答案生成

DeepSeek会：

1. 基于检索到的知识库内容
2. 结合其预训练的医学知识
3. 生成专业、准确、易懂的答案
4. 保持结构化和逻辑性

## 📊 系统架构

```
用户提问
    ↓
Web界面 / API
    ↓
DeepSeek问答系统
    ↓
├─ 知识库检索（可选）
│   ├─ 向量检索
│   └─ 关键词检索
│
└─ DeepSeek API调用
    ├─ 系统提示词
    ├─ 用户问题
    └─ 知识上下文
    ↓
AI生成的答案
    ↓
返回给用户
```

## 💡 使用建议

### 1. 首次使用

1. 先测试API连接：`[6] 测试DeepSeek API连接`
2. 确认API正常后启动服务：`[5] 启动DeepSeek智能问答系统`
3. 在Web界面中测试问答功能

### 2. 优化答案质量

- **使用知识库检索**: 启用"使用知识库"选项可以获得更准确的答案
- **调整检索数量**: 根据问题复杂度调整top_k参数
- **优化问题表述**: 清晰、具体的问题会得到更好的答案

### 3. 性能优化

- **网络连接**: 确保网络连接稳定，API调用需要网络
- **API配额**: 注意DeepSeek API的使用配额
- **缓存机制**: 相同问题可以考虑缓存答案

## ⚠️ 注意事项

### API相关

1. **API Key安全**
   - 不要将API Key提交到代码仓库
   - 定期更换API Key
   - 监控API使用情况

2. **网络要求**
   - 需要稳定的网络连接
   - API调用可能有延迟
   - 建议设置合理的超时时间

3. **费用控制**
   - 注意API调用的token使用量
   - 合理设置max_tokens参数
   - 监控API费用

### 知识库相关

1. **数据质量**
   - 确保知识库数据准确
   - 定期更新知识库内容
   - 去除重复和错误数据

2. **检索效果**
   - 向量检索基于hash向量，可能不如专业embedding模型
   - 关键词检索依赖于分词质量
   - 可以调整检索权重优化效果

## 🔧 故障排除

### 常见问题

#### 1. API调用失败

**问题**: 提示API调用失败

**解决方案**:
- 检查网络连接
- 验证API Key是否正确
- 确认API地址是否正确
- 查看API配额是否用完

#### 2. 答案质量不佳

**问题**: 答案不准确或不相关

**解决方案**:
- 启用知识库检索
- 调整检索参数（top_k）
- 优化问题表述
- 检查知识库数据质量

#### 3. 响应速度慢

**问题**: 答案生成时间过长

**解决方案**:
- 检查网络连接速度
- 减少max_tokens参数
- 降低temperature参数
- 考虑使用缓存

#### 4. Web界面无法访问

**问题**: 无法打开Web界面

**解决方案**:
- 确认服务已启动
- 检查端口5001是否被占用
- 尝试使用127.0.0.1:5001
- 检查防火墙设置

## 📞 技术支持

如果遇到问题，可以：

1. **查看日志**: 检查系统日志了解详细错误信息
2. **测试API**: 使用测试脚本验证API连接
3. **检查状态**: 查看系统状态了解配置信息
4. **查看文档**: 参考本文档和API文档

## 🔄 版本更新

### 当前版本: v1.0.0

**功能特性**:
- ✅ DeepSeek API集成
- ✅ 知识库检索模拟
- ✅ Web界面
- ✅ RESTful API
- ✅ 混合检索支持
- ✅ 批量问答支持

### 未来计划

- [ ] 支持流式响应
- [ ] 添加对话历史
- [ ] 支持多轮对话
- [ ] 添加答案评分
- [ ] 支持自定义提示词
- [ ] 添加使用统计

## 📄 许可证

本项目仅供学习和研究使用。

---

**最后更新**: 2026-02-15  
**Python版本**: 3.10  
**DeepSeek模型**: deepseek-chat
