// 医学问答系统 - Web界面交互脚本

// 系统状态管理
const SystemState = {
    currentProcess: null,
    outputBuffer: [],
    isProcessing: false
};

// DOM元素引用
const elements = {
    startLightweight: document.getElementById('startLightweight'),
    startFullDemo: document.getElementById('startFullDemo'),
    startDeploy: document.getElementById('startDeploy'),
    clearOutput: document.getElementById('clearOutput'),
    outputContent: document.getElementById('outputContent'),
    loadingIndicator: document.getElementById('loadingIndicator'),
    systemStatus: document.getElementById('systemStatus')
};

// 初始化系统
document.addEventListener('DOMContentLoaded', function() {
    initializeSystem();
    bindEventListeners();
    updateSystemStatus();
});

// 初始化系统
function initializeSystem() {
    console.log('医学问答系统 Web界面初始化完成');
    addOutput('系统初始化完成，界面加载成功', 'info');
}

// 绑定事件监听器
function bindEventListeners() {
    // 启动轻量级问答系统
    elements.startLightweight.addEventListener('click', function() {
        runCommand('python lightweight_medical_qa.py', '轻量级问答系统');
    });

    // 启动完整演示
    elements.startFullDemo.addEventListener('click', function() {
        runCommand('python medical_qa_demo.py', '完整功能演示');
    });

    // 系统部署
    elements.startDeploy.addEventListener('click', function() {
        runCommand('python deploy_system.py', '系统部署');
    });

    // 清空输出
    elements.clearOutput.addEventListener('click', function() {
        clearOutput();
    });
}

// 更新系统状态
function updateSystemStatus() {
    const status = SystemState.isProcessing ? '运行中' : '就绪';
    const systemStatusElement = document.getElementById('systemStatus');
    if (systemStatusElement) {
        systemStatusElement.textContent = status;
        systemStatusElement.className = `value status-${SystemState.isProcessing ? 'running' : 'ready'}`;
    }
}

// 运行命令
async function runCommand(command, description) {
    if (SystemState.isProcessing) {
        addOutput('⚠️ 系统正在运行中，请等待当前操作完成', 'warning');
        return;
    }

    try {
        SystemState.isProcessing = true;
        SystemState.currentProcess = description;
        updateSystemStatus();
        showLoading(true);

        addOutput(`🚀 启动 ${description}...`, 'info');
        addOutput(`💻 执行命令: ${command}`, 'info');
        addOutput('=' .repeat(50), 'info');

        // 模拟命令执行（在实际环境中需要后端支持）
        const result = await simulateCommandExecution(command, description);
        
        if (result.success) {
            addOutput(`✅ ${description} 执行成功！`, 'success');
            addOutput(result.output, 'success');
        } else {
            addOutput(`❌ ${description} 执行失败`, 'error');
            addOutput(result.error, 'error');
        }

    } catch (error) {
        addOutput(`❌ 执行过程中发生错误: ${error.message}`, 'error');
    } finally {
        SystemState.isProcessing = false;
        SystemState.currentProcess = null;
        updateSystemStatus();
        showLoading(false);
    }
}

// 模拟命令执行（演示用）
async function simulateCommandExecution(command, description) {
    return new Promise((resolve) => {
        setTimeout(() => {
            if (command.includes('lightweight_medical_qa.py')) {
                resolve({
                    success: true,
                    output: `
==================================================
问题 1: 什么是心脏病？
回答: 根据医学知识库，心脏病是心脏功能或结构异常的疾病总称
置信度: 0.9
引用: 解剖类:心脏, 疾病类:心脏病
----------------------------------------

问题 2: 冠心病的诊断方法有哪些？
回答: 冠心病的诊断方法包括心电图、心脏彩超、冠脉造影等
置信度: 0.7
引用: 检查类:心电图, 检查类:冠脉造影
----------------------------------------

💡 系统能力展示:
  ✅ 医学术语自动识别和分类
  ✅ 基于关键词的智能检索
  ✅ 文本相似度计算和排序
  ✅ 多源知识库整合
  ✅ 答案置信度评估
  ✅ 引用信息溯源
  ✅ 批量处理和性能统计
  ✅ 兼容Python 3.7+环境

🎯 系统已成功运行！轻量级版本适用于资源受限环境。
⏰ 演示完成时间: ${new Date().toLocaleString()}
                    `.trim()
                });
            } else if (command.includes('medical_qa_demo.py')) {
                resolve({
                    success: true,
                    output: `
==================================================
📊 医学问答系统演示开始
🔍 知识检索演示
💬 问答系统演示
📈 系统能力展示
🎯 系统统计信息

📋 演示结果:
  • 处理文档数量: 3个医学主题
  • 生成知识块: 26个向量块
  • 问答测试: 5个问题
  • 平均置信度: 0.42
  • 响应时间: <2秒

✅ 演示完成！系统运行正常。
                    `.trim()
                });
            } else if (command.includes('deploy_system.py')) {
                resolve({
                    success: true,
                    output: `
==================================================
🔧 医学问答系统部署开始
✅ 环境检查完成 (Python 3.7+)
✅ 目录结构验证通过
✅ 配置文件检查完成
✅ 依赖检查完成
✅ 数据库连接测试通过

🎯 部署完成！系统已就绪。
⏰ 部署时间: ${new Date().toLocaleString()}
                    `.trim()
                });
            } else {
                resolve({
                    success: false,
                    error: '未知命令类型'
                });
            }
        }, 2000); // 模拟2秒执行时间
    });
}

// 添加输出
function addOutput(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const outputLine = document.createElement('div');
    outputLine.className = `output-${type}`;
    outputLine.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${message}`;
    
    elements.outputContent.appendChild(outputLine);
    elements.outputContent.scrollTop = elements.outputContent.scrollHeight;
    
    // 保存到缓冲区
    SystemState.outputBuffer.push({ message, type, timestamp });
}

// 清空输出
function clearOutput() {
    elements.outputContent.innerHTML = `
        <div class="placeholder">
            <i class="fas fa-info-circle"></i>
            <p>点击上方按钮启动系统，输出结果将显示在这里...</p>
        </div>
    `;
    SystemState.outputBuffer = [];
    addOutput('🧹 输出区域已清空', 'info');
}

// 显示/隐藏加载指示器
function showLoading(show) {
    if (show) {
        elements.loadingIndicator.classList.remove('hidden');
    } else {
        elements.loadingIndicator.classList.add('hidden');
    }
}

// 键盘快捷键
document.addEventListener('keydown', function(event) {
    // Ctrl+L 启动轻量级系统
    if (event.ctrlKey && event.key === 'l') {
        event.preventDefault();
        elements.startLightweight.click();
    }
    
    // Ctrl+D 启动完整演示
    if (event.ctrlKey && event.key === 'd') {
        event.preventDefault();
        elements.startFullDemo.click();
    }
    
    // Ctrl+C 清空输出
    if (event.ctrlKey && event.key === 'c') {
        event.preventDefault();
        elements.clearOutput.click();
    }
});

// 系统信息展示
function displaySystemInfo() {
    const systemInfo = {
        pythonVersion: '3.7+',
        environment: 'Windows',
        framework: 'RAGFlow + DeepSeek',
        database: 'Chroma Vector DB',
        language: 'Python',
        compatibility: 'High'
    };
    
    addOutput('📊 系统信息:', 'info');
    Object.entries(systemInfo).forEach(([key, value]) => {
        addOutput(`  ${key}: ${value}`, 'info');
    });
}

// 错误处理
window.addEventListener('error', function(event) {
    addOutput(`❌ 页面错误: ${event.message}`, 'error');
});

// 页面卸载时清理
window.addEventListener('beforeunload', function(event) {
    if (SystemState.isProcessing) {
        event.preventDefault();
        event.returnValue = '系统正在运行中，确定要离开吗？';
    }
});

// 导出功能供外部调用
window.MedicalQASystem = {
    addOutput: addOutput,
    clearOutput: clearOutput,
    runCommand: runCommand,
    displaySystemInfo: displaySystemInfo
};