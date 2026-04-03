use pyo3::prelude::*;
use tiktoken_rs::cl100k_base;
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[pyclass(module = "bdm_rust", get_all, set_all)]
#[derive(Clone)]
pub struct DistilledMemory {
    pub memory_id: String,
    pub source_chunk_id: String,
    pub structured_summary: PyObject,
    pub entities: Vec<String>,
    pub decisions: Vec<String>,
    pub actions: Vec<String>,
    pub constraints: Vec<String>,
    pub preferences: Vec<String>,
    pub code_snippets: Vec<String>,
    pub important_facts: Vec<String>,
    pub compression_ratio: f64,
    pub fidelity_score: f64,
    pub generation_cost: usize,
    pub embedding: Vec<f64>,
}

#[pymethods]
impl DistilledMemory {
    #[new]
    #[pyo3(signature = (source_chunk_id, structured_summary, entities=None, decisions=None, actions=None, constraints=None, preferences=None, code_snippets=None, important_facts=None, compression_ratio=0.0, fidelity_score=1.0, generation_cost=0, embedding=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        source_chunk_id: String,
        structured_summary: PyObject,
        entities: Option<Vec<String>>,
        decisions: Option<Vec<String>>,
        actions: Option<Vec<String>>,
        constraints: Option<Vec<String>>,
        preferences: Option<Vec<String>>,
        code_snippets: Option<Vec<String>>,
        important_facts: Option<Vec<String>>,
        compression_ratio: f64,
        fidelity_score: f64,
        generation_cost: usize,
        embedding: Option<Vec<f64>>,
    ) -> Self {
        DistilledMemory {
            memory_id: Uuid::new_v4().to_string(),
            source_chunk_id,
            structured_summary,
            entities: entities.unwrap_or_default(),
            decisions: decisions.unwrap_or_default(),
            actions: actions.unwrap_or_default(),
            constraints: constraints.unwrap_or_default(),
            preferences: preferences.unwrap_or_default(),
            code_snippets: code_snippets.unwrap_or_default(),
            important_facts: important_facts.unwrap_or_default(),
            compression_ratio,
            fidelity_score,
            generation_cost,
            embedding: embedding.unwrap_or_default(),
        }
    }
}

#[pyclass(module = "bdm_rust", get_all, set_all)]
#[derive(Clone)]
pub struct MemoryChunk {
    pub chunk_id: String,
    pub raw_text: String,
    pub tokens: usize,
    pub timestamp: DateTime<Utc>,
    pub semantic_boundary: bool,
    pub status: String,
    pub access_count: usize,
    pub last_accessed: DateTime<Utc>,
    pub forgetting_rate: f64,
    pub recovery_potential: f64,
    pub importance_score: f64,
    pub related_chunks: Vec<String>,
    pub distilled_version: Option<Py<DistilledMemory>>,
}

#[pymethods]
impl MemoryChunk {
    #[new]
    #[pyo3(signature = (raw_text, tokens, semantic_boundary=false))]
    fn new(raw_text: String, tokens: usize, semantic_boundary: bool) -> Self {
        let now = Utc::now();
        MemoryChunk {
            chunk_id: Uuid::new_v4().to_string(),
            raw_text,
            tokens,
            timestamp: now,
            semantic_boundary,
            status: "active".to_string(),
            access_count: 0,
            last_accessed: now,
            forgetting_rate: 0.01,
            recovery_potential: 1.0,
            importance_score: 1.0,
            related_chunks: Vec::new(),
            distilled_version: None,
        }
    }

    fn update_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = Utc::now();
    }
}

/// 智能分块算法的配置
#[pyclass(module = "bdm_rust", get_all, set_all)]
#[derive(Clone)]
pub struct ChunkingConfig {
    pub max_tokens: usize,
    pub overlap_tokens: usize,
}

#[pymethods]
impl ChunkingConfig {
    #[new]
    #[pyo3(signature = (max_tokens=20000, overlap_tokens=100))]
    fn new(max_tokens: usize, overlap_tokens: usize) -> Self {
        ChunkingConfig {
            max_tokens,
            overlap_tokens,
        }
    }
}

/// 模拟的语义边界检测（简单的句号、换行符检测）
fn detect_semantic_boundary(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.ends_with('.') || trimmed.ends_with('。') || trimmed.ends_with('!') || trimmed.ends_with('！') || trimmed.ends_with('?') || trimmed.ends_with('？') || trimmed.ends_with('\n')
}

/// 智能分块算法实现
#[pyfunction]
#[pyo3(signature = (conversation_stream, config))]
fn intelligent_chunking(conversation_stream: Vec<String>, config: ChunkingConfig) -> PyResult<Vec<MemoryChunk>> {
    let mut chunks = Vec::new();
    let mut current_chunk_texts = Vec::new();
    let mut current_token_count = 0;

    let bpe = cl100k_base().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    for utterance in conversation_stream {
        let utterance_tokens = bpe.encode_with_special_tokens(&utterance).len();
        
        let is_boundary = detect_semantic_boundary(&utterance);
        
        // 分块条件：达到长度上限，或者在有一定内容积累的情况下遇到了语义边界
        if current_token_count + utterance_tokens > config.max_tokens 
           || (is_boundary && current_token_count > config.max_tokens / 2) {
            
            if !current_chunk_texts.is_empty() {
                let raw_text = current_chunk_texts.join("\n");
                chunks.push(MemoryChunk::new(raw_text, current_token_count, is_boundary));
                
                // 处理重叠逻辑 (Overlap) - 保留最后一条发言作为上下文重叠
                current_chunk_texts.clear();
                current_chunk_texts.push(utterance.clone());
                current_token_count = utterance_tokens;
                continue;
            }
        }
        
        current_chunk_texts.push(utterance);
        current_token_count += utterance_tokens;
    }
    
    // 处理最后剩余的块
    if !current_chunk_texts.is_empty() {
        let raw_text = current_chunk_texts.join("\n");
        chunks.push(MemoryChunk::new(raw_text, current_token_count, true));
    }

    Ok(chunks)
}

/// 动态遗忘算法
#[pyclass(module = "bdm_rust", get_all, set_all)]
#[derive(Clone)]
pub struct DynamicForgetting {
    pub base_forgetting_rate: f64,
    pub recovery_threshold: f64,
}

#[pymethods]
impl DynamicForgetting {
    #[new]
    #[pyo3(signature = (base_forgetting_rate=0.01, recovery_threshold=0.7))]
    fn new(base_forgetting_rate: f64, recovery_threshold: f64) -> Self {
        DynamicForgetting {
            base_forgetting_rate,
            recovery_threshold,
        }
    }

    /// 计算访问模式影响因子 (0.0 - 1.0)
    fn calculate_access_pattern(&self, chunk: &MemoryChunk) -> f64 {
        // 简单的对数平滑，访问次数越多，影响因子越趋近于 1.0 (抑制遗忘)
        let access = chunk.access_count as f64;
        if access == 0.0 {
            0.0
        } else {
            (access.ln() / 10.0).clamp(0.0, 0.9)
        }
    }

    /// 计算关联激活度 (Graph-like spreading activation)
    fn calculate_related_activation(&self, chunk: &MemoryChunk) -> f64 {
        // 在真实图数据库中，这里会根据 related_chunks 的活跃度加权求和。
        // MVP 中，我们模拟：如果有任何关联记忆，则给一个基础的激活乘数，防止被过快遗忘。
        if chunk.related_chunks.is_empty() {
            1.0
        } else {
            1.2 // 有关联的记忆，恢复潜力提升 20%
        }
    }

    /// 更新记忆块的状态
    fn update_memory_state(&self, mut chunk: MemoryChunk) -> MemoryChunk {
        let now = Utc::now();
        let time_passed = now.signed_duration_since(chunk.last_accessed);
        let days_passed = time_passed.num_days() as f64;
        
        // 计算年衰减 (0.0 - 1.0)
        let time_decay = (days_passed / 365.0).clamp(0.0, 1.0);
        
        let access_pattern = self.calculate_access_pattern(&chunk);
        
        // 更新遗忘率
        chunk.forgetting_rate = self.base_forgetting_rate 
            * (1.0 + time_decay) 
            * (1.0 - access_pattern);
            
        // 更新恢复潜力
        let related_activation = self.calculate_related_activation(&chunk); 
        chunk.recovery_potential = chunk.importance_score 
            * (1.0 - time_decay) 
            * related_activation;
            
        // 状态转移逻辑
        if chunk.status == "active" {
            if chunk.forgetting_rate > self.recovery_threshold {
                chunk.status = "latent".to_string();
            }
        } else if chunk.status == "latent" {
            if chunk.recovery_potential > chunk.forgetting_rate {
                chunk.status = "active".to_string();
            } else if chunk.forgetting_rate > 0.95 {
                chunk.status = "archived".to_string();
            }
        }
        
        chunk
    }
}

/// Count tokens in a given string using cl100k_base (OpenAI/Deepseek approximation)
#[pyfunction]
fn count_tokens(text: &str) -> PyResult<usize> {
    let bpe = cl100k_base().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let tokens = bpe.encode_with_special_tokens(text);
    Ok(tokens.len())
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn bdm_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DistilledMemory>()?;
    m.add_class::<MemoryChunk>()?;
    m.add_class::<ChunkingConfig>()?;
    m.add_class::<DynamicForgetting>()?;
    m.add_function(wrap_pyfunction!(intelligent_chunking, m)?)?;
    m.add_function(wrap_pyfunction!(count_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
