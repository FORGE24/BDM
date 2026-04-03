
# MLED (Machine Life Evolutionary Distilling)

MLED 是一套融合了**认知科学**与**进化算法**的 AI 机器生命进化蒸馏系统。它不仅是一个聊天机器人，更是一个具备“类生命特征”的认知引擎。系统通过 **Rust (PyO3)** 实现高性能的底层记忆与分块计算，并结合 **Python** 驱动顶层业务逻辑与大模型 (Deepseek) 的蒸馏过程。

## 🌟 核心特性

- 🧠 **级联记忆自蒸馏 (Cascading Self-Distillation)**：系统会在对话达到语义边界时自动触发分块。每次分块不仅提取当前流的核心实体、决策和事实，还会按 50% 的压缩率继承上一轮的记忆精华，保证了长期话题的绝对连贯性。
- 🕸️ **混合记忆检索 (Hybrid Memory Retrieval)**：基于 SQLite 和倒排关键字匹配，支持跨越时间周期的记忆提取，自动组装为系统上下文。
- ⏳ **动态遗忘算法 (Dynamic Forgetting)**：所有存入的记忆块（Memory Chunk）会根据时间衰减和访问频率（对数平滑）计算遗忘率，并在 `active`、`latent`、`archived` 三种状态中自动流转。
- 🧬 **自进化引擎 (Evolutionary Optimizer)**：根据系统的任务成功率、记忆保真度等指标计算“适应度”，当适应度停滞时，会自动通过变异算法生成并应用新一代的超参数配置（如分块大小、遗忘阈值等）。
- 🛡️ **硬编码安全护栏 (Safety Guardrail)**：在输入处理的第一道防线进行安全词汇和意图的强力拦截。
- 🦀 **Rust 底层加速**：核心的分块算法、Token 计数器、动态遗忘状态计算均使用 Rust 编写，通过 Maturin 打包为极速 Python 扩展模块。

## 🛠️ 技术栈

- **底层引擎**: Rust, PyO3, Maturin
- **上层逻辑**: Python 3.8+
- **大模型支持**: Deepseek API (OpenAI 兼容客户端)
- **数据库**: SQLite (SQLAlchemy)
- **数据校验**: Pydantic

## 🚀 快速开始

### 1. 环境准备
确保你的系统已安装了 **Python 3.8+** 以及 **Rust 工具链** (Cargo)。

### 2. 克隆与安装
```bash
# 克隆仓库
git clone https://github.com/FORGE24/BDM.git
cd BDM

# 创建并激活虚拟环境 (Windows)
python -m venv .venv
.\.venv\Scripts\activate

# 安装依赖
pip install maturin sqlalchemy openai python-dotenv pydantic

# 编译并安装 Rust 扩展
maturin develop
```

### 3. 配置 API Key
在项目根目录下创建一个 `.env` 文件（或修改 `.env.example`）：
```env
DEEPSEEK_API_KEY=sk-你的DeepseekAPIKey
```

### 4. 运行交互终端
```bash
python main.py
```

## 🎮 控制台指令说明

在 `main.py` 交互式终端中，你可以正常输入任何对话，也可以使用以下特殊指令触发系统的机器生命行为：

- `flush`：主动截断当前的对话流，强制系统立即执行一次**记忆蒸馏**并写入数据库。
- `forget`：模拟后台的时间推移，执行一次**动态遗忘周期**，重新计算所有记忆块的衰减率并更新状态。
- `evolve`：评估系统当前的表现，执行一次**自进化周期**。如果表现欠佳，系统会自我变异并应用新一代配置。
- `quit`：安全退出系统。

## 📁 目录结构

```text
BDM/
├── src/
│   └── lib.rs             # Rust 底层引擎：核心数据结构与算法
├── database.py            # SQLite 数据库与 SQLAlchemy 模型映射
├── distiller.py           # Deepseek API 调用与结构化蒸馏逻辑
├── evolution.py           # 进化算法、适应度评估与超参数变异
├── intent.py              # 用户对话意图与任务完成度识别
├── main.py                # DialogManager，系统主循环入口
├── memory.py              # Python 端的记忆对象结构透传
├── retriever.py           # 基于关键字打分的历史记忆混合检索器
├── safety.py              # 第一道安全防线（不可进化安全边界）
├── Cargo.toml             # Rust 依赖配置
├── pyproject.toml         # Maturin 构建配置
└── BDM介绍.md             # 原始系统架构设计文档
```

## 📝 许可证

GPL2
