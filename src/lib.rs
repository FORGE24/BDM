use pyo3::prelude::*;
use tiktoken_rs::cl100k_base;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::collections::{HashMap, VecDeque};

mod advanced;

// ============= 数据结构定义 =============

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
    
    // DAG 因果链接 (Causal Linking)
    pub parent_nodes: Vec<String>,
    
    // 热度衰减权重
    pub heat_score: f64,
    
    // 2MNSAS 达尔文循环进化属性 (Darwinian Evolution Traits)
    pub context_tag: String,      // 隔离上下文 (e.g., "math", "story", "reasoning")
    pub fitness: f64,             // 适应度 (低于阈值会被淘汰)
    pub success_rate: f64,        // 成功调用率
    pub usage_count: usize,       // 历史调用次数
}

#[pymethods]
impl DistilledMemory {
    #[new]
    #[pyo3(signature = (source_chunk_id, structured_summary, entities=None, decisions=None, actions=None, constraints=None, preferences=None, code_snippets=None, important_facts=None, compression_ratio=0.0, fidelity_score=1.0, generation_cost=0, embedding=None, parent_nodes=None, heat_score=1.0, context_tag="general", fitness=0.5, success_rate=0.5, usage_count=0))]
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
        parent_nodes: Option<Vec<String>>,
        heat_score: f64,
        context_tag: &str,
        fitness: f64,
        success_rate: f64,
        usage_count: usize,
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
            parent_nodes: parent_nodes.unwrap_or_default(),
            heat_score,
            context_tag: context_tag.to_string(),
            fitness,
            success_rate,
            usage_count,
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

// ============= DAG 结构和算法 =============

/// DAG 节点表示 (节点 = 蒸馏记忆块)
#[derive(Clone, Debug)]
pub struct DAGNode {
    pub memory_id: String,
    pub source_chunk_id: String,
    pub parent_nodes: Vec<String>,  // 父节点 ID 列表
    pub heat_score: f64,             // 热度评分
    pub access_count: usize,         // 访问次数
    pub token_length: usize,         // 内容长度 (token)
    pub distilled_content: String,   // 蒸馏摘要简单表示
}

/// DAG 图结构 - 用于拓扑检索
#[pyclass(module = "bdm_rust")]
pub struct MemoryDAG {
    pub nodes: HashMap<String, DAGNode>,
    pub adjacency_list: HashMap<String, Vec<String>>, // 正向邻接表 (child)
    pub reverse_adjacency: HashMap<String, Vec<String>>, // 反向邻接表 (parent)
}

#[pymethods]
impl MemoryDAG {
    #[new]
    fn new() -> Self {
        MemoryDAG {
            nodes: HashMap::new(),
            adjacency_list: HashMap::new(),
            reverse_adjacency: HashMap::new(),
        }
    }

    /// 添加节点
    fn add_node(&mut self, memory_id: String, parent_nodes: Vec<String>, heat_score: f64, access_count: usize, token_length: usize, distilled_content: String) {
        let node = DAGNode {
            memory_id: memory_id.clone(),
            source_chunk_id: memory_id.clone(),
            parent_nodes: parent_nodes.clone(),
            heat_score,
            access_count,
            token_length,
            distilled_content,
        };
        
        self.nodes.insert(memory_id.clone(), node);
        
        // 构建邻接表
        self.adjacency_list.entry(memory_id.clone()).or_insert_with(Vec::new);
        self.reverse_adjacency.entry(memory_id.clone()).or_insert_with(Vec::new);
        
        // 建立父子关系
        for parent_id in parent_nodes.iter() {
            self.adjacency_list
                .entry(parent_id.clone())
                .or_insert_with(Vec::new)
                .push(memory_id.clone());
            
            self.reverse_adjacency
                .entry(memory_id.clone())
                .or_insert_with(Vec::new)
                .push(parent_id.clone());
        }
    }

    /// 拓扑启发式检索：给定目标节点，沿着 DAG 向上回溯，返回上下文链
    /// 返回格式: [(node_id, depth, distance, token_allocation), ...]
    fn heuristic_retrieval(&self, target_node_id: String, max_depth: i32, max_tokens: i32) -> Vec<(String, i32, i32, i32)> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut queue = VecDeque::new();
        
        queue.push_back((target_node_id.clone(), 0, 0)); // (node_id, depth, distance)
        visited.insert(target_node_id.clone());
        
        // BFS 向上回溯
        while let Some((node_id, depth, distance)) = queue.pop_front() {
            if depth > max_depth {
                continue;
            }
            
            if let Some(node) = self.nodes.get(&node_id) {
                // Token 分配策略：远疏近亲
                // 深度 0 (当前): 70% tokens
                // 深度 1 (父): 20% tokens
                // 深度 2+ (祖先): 10% tokens
                let token_allocation = match depth {
                    0 => (max_tokens as f64 * 0.7) as i32,
                    1 => (max_tokens as f64 * 0.2) as i32,
                    _ => (max_tokens as f64 * 0.1) as i32,
                };
                
                result.push((node_id.clone(), depth, distance, token_allocation));
                
                // 加入父节点到队列
                if let Some(parents) = self.reverse_adjacency.get(&node_id) {
                    for parent_id in parents.iter() {
                        if !visited.contains(parent_id) {
                            visited.insert(parent_id.clone());
                            queue.push_back((parent_id.clone(), depth + 1, distance + 1));
                        }
                    }
                }
            }
        }
        
        result
    }
    
    /// 获取节点的完整上下文链 (用于符号协议输出)
    /// 返回格式: "Ref[A->C->D]: content"
    fn get_context_chain(&self, target_node_id: String) -> String {
        let mut chain = vec![target_node_id.clone()];
        let mut current = &target_node_id;
        
        // 向上回溯到根节点
        let mut iterations = 0;
        while iterations < 100 {
            if let Some(parents) = self.reverse_adjacency.get(current) {
                if !parents.is_empty() {
                    current = &parents[0]; // 取第一个父节点
                    chain.insert(0, current.clone());
                } else {
                    break;
                }
            } else {
                break;
            }
            iterations += 1;
        }
        
        // 格式化为符号协议
        let chain_str = chain.join("->");
        format!("Ref[{}]", chain_str)
    }
}

// ============= 热度衰减算法 (Pagerank-like with Heat Decay) =============

/// 热度衰减算法 - 给节点计算热度分数
#[pyclass(module = "bdm_rust")]
pub struct HeatDecayEngine {
    pub decay_factor: f64,         // 衰减因子 (0.0 - 1.0)
    pub recency_weight: f64,       // 最近访问权重
    pub access_freq_weight: f64,   // 访问频率权重
    pub relation_weight: f64,      // 关系权重
}

#[pymethods]
impl HeatDecayEngine {
    #[new]
    #[pyo3(signature = (decay_factor=0.95, recency_weight=0.4, access_freq_weight=0.3, relation_weight=0.3))]
    fn new(decay_factor: f64, recency_weight: f64, access_freq_weight: f64, relation_weight: f64) -> Self {
        HeatDecayEngine {
            decay_factor,
            recency_weight,
            access_freq_weight,
            relation_weight,
        }
    }

    /// 计算单个节点的热度分数
    /// 输入: (access_count, days_since_access, parent_count, is_referenced)
    /// 输出: 热度分数 (0.0 - 1.0)
    fn calculate_heat_score(&self, access_count: usize, days_since_access: f64, parent_count: usize, is_referenced: bool) -> f64 {
        // 最近访问权重 (时间衰减)
        let recency_score = (1.0 / (1.0 + days_since_access / 7.0)).clamp(0.0, 1.0); // 7天为衰减半周期
        
        // 访问频率权重 (对数平滑)
        let access_score = ((access_count as f64).ln() / 10.0).clamp(0.0, 1.0);
        
        // 关系权重 (父节点或被引用则提升)
        let relation_score = if parent_count > 0 || is_referenced { 0.8 } else { 0.3 };
        
        let heat = (self.recency_weight * recency_score) +
                   (self.access_freq_weight * access_score) +
                   (self.relation_weight * relation_score);
        
        heat.clamp(0.0, 1.0)
    }

    /// Pagerank-like 热度传播算法
    /// 返回需要剪枝的节点 ID 列表 (热度低于阈值)
    fn identify_pruning_candidates(&self, node_scores: Vec<(String, f64)>, pruning_threshold: f64) -> Vec<String> {
        node_scores
            .into_iter()
            .filter(|(_, score)| *score < pruning_threshold)
            .map(|(id, _)| id)
            .collect()
    }

    /// 热度衰减迭代 (每次调用表示一个时间步)
    fn decay_iteration(&self, heat_scores: Vec<f64>) -> Vec<f64> {
        heat_scores
            .into_iter()
            .map(|h| h * self.decay_factor)
            .collect()
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
    // Core structures
    m.add_class::<DistilledMemory>()?;
    m.add_class::<MemoryChunk>()?;
    m.add_class::<ChunkingConfig>()?;
    m.add_class::<DynamicForgetting>()?;
    m.add_class::<MemoryDAG>()?;
    m.add_class::<HeatDecayEngine>()?;
    
    // Advanced features - Predictive Coding & Sparse Impulse
    m.add_class::<advanced::NeuralSpike>()?;
    m.add_class::<advanced::SurpriseFilter>()?;
    
    // Advanced features - Local Consolidation
    m.add_class::<advanced::ConsolidatedBlock>()?;
    m.add_class::<advanced::LocalConsolidationEngine>()?;
    
    // Advanced features - MoE World Model
    m.add_class::<advanced::MoERouter>()?;
    m.add_class::<advanced::WorldModelExecutor>()?;
    
    // Functions
    m.add_function(wrap_pyfunction!(intelligent_chunking, m)?)?;
    m.add_function(wrap_pyfunction!(count_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
