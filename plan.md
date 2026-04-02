# 使用 Rust 实现蒸馏记忆结构的计划

## 背景与目标
根据《BDM介绍.md》的设计，系统应当具备极高的数据处理效率。目前 `MemoryChunk` 与 `DistilledMemory` 是由 Python (Pydantic) 实现的。为了提升性能，并在未来支持更底层的密集记忆检索计算，我们将把核心数据结构全部迁移到 Rust 层，然后通过 `PyO3` 暴露给 Python 调用。

## 实施步骤

1. **配置 Rust 依赖**：
   - 引入 `uuid` 用于生成唯一的记忆块 ID。
   - 引入 `chrono` 并开启 `pyo3` 的 `chrono` 特性，以支持将 Rust 原生时间戳无缝转换为 Python 的 `datetime` 对象。
2. **在 Rust 中定义 `DistilledMemory`**：
   - 使用 `#[pyclass]` 宏将其包装为 Python 可识别的类。
   - 提供 `#[new]` 构造函数并声明 `#[pyo3(signature = (...))]`，以便能用关键字参数进行实例化。
   - 暴露所有的结构化列表字段（实体、行动、偏好、代码片段等）。
3. **在 Rust 中定义 `MemoryChunk`**：
   - 使用 `#[pyclass]` 包装。
   - 提供自动生成 ID 和当前时间戳的构造能力，以及 `update_access()` 访问次数更新方法。
   - `distilled_version` 属性将直接存储对 `DistilledMemory` 实例的引用 (`Option<Py<DistilledMemory>>`)。
4. **编译与替换**：
   - 执行 `maturin develop` 进行编译构建。
   - 修改 Python 端的 `memory.py`，将原来的 Pydantic 实现替换为从 `bdm_rust` 原生导入的模块，做到对上层调用完全透明。
5. **系统测试**：
   - 运行 `python main.py`，确保 Python 端的对话管理器、蒸馏引擎，以及 SQLAlchemy 能够完美地读写这些来自 Rust 层的对象。
