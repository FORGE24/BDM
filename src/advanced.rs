// Advanced BDM features: Predictive Coding, Local Consolidation, MoE World Model
// File: src/advanced.rs

use std::collections::{HashMap, VecDeque};
use pyo3::prelude::*;

// ============================================================================
// 1. PREDICTIVE CODING & SPARSE IMPULSE
// ============================================================================

    /// 神经元激活特征 - 记录内存访问的"稀疏脉冲"
#[pyclass(module = "bdm_rust")]
#[derive(Clone, Debug)]
pub struct NeuralSpike {
    #[pyo3(get)]
    pub node_id: String,
    #[pyo3(get)]
    pub surprise_score: f64,          // 0.0-1.0: 惊奇度 (预测误差)
    #[pyo3(get)]
    pub spike_magnitude: f64,         // 0.0-1.0: 脉冲强度
    #[pyo3(get)]
    pub timestamp: i64,               // Unix 时间戳
    #[pyo3(get)]
    pub activation_type: String,      // "routine", "novel", "error"
    #[pyo3(get)]
    pub fiber_channel: String,        // 脑干光纤通道 (例如: "logic_pathway", "memory_bus")
}

#[pymethods]
impl NeuralSpike {
    #[new]
    #[pyo3(signature = (node_id, surprise_score, spike_magnitude=1.0, timestamp=0, activation_type="routine", fiber_channel="default_bus"))]
    fn new(
        node_id: String,
        surprise_score: f64,
        spike_magnitude: f64,
        timestamp: i64,
        activation_type: &str,
        fiber_channel: &str,
    ) -> Self {
        NeuralSpike {
            node_id,
            surprise_score: surprise_score.clamp(0.0, 1.0),
            spike_magnitude: spike_magnitude.clamp(0.0, 1.0),
            timestamp,
            activation_type: activation_type.to_string(),
            fiber_channel: fiber_channel.to_string(),
        }
    }
    
    /// Getter: 获取节点ID
    #[getter]
    fn get_node_id(&self) -> String {
        self.node_id.clone()
    }
    
    /// Getter: 获取惊奇度
    #[getter]
    fn get_surprise_score(&self) -> f64 {
        self.surprise_score
    }
    
    /// Getter: 获取脉冲强度
    #[getter]
    fn get_spike_magnitude(&self) -> f64 {
        self.spike_magnitude
    }
    
    /// Getter: 获取激活类型
    #[getter] 
    fn get_activation_type(&self) -> String {
        self.activation_type.clone()
    }
}

/// 预测编码系统 - Surprise Filter
#[pyclass(module = "bdm_rust")]
pub struct SurpriseFilter {
    // 预测缓存：node_id -> (expected_embedding, confidence)
    prediction_cache: HashMap<String, (Vec<f64>, f64)>,
    
    // 运行窗口的平均惊奇度
    running_surprise: f64,
    
    // 历史脉冲记录
    spike_history: VecDeque<NeuralSpike>,
    
    // 生命值 (Vitality) 机制：用于触发恐惧脉冲
    #[pyo3(get)]
    pub vitality: f64,                // 0.0-1.0，随时间衰减
    #[pyo3(get)]
    pub vitality_decay_rate: f64,     // 生命值衰减率
    #[pyo3(get)]
    pub fear_threshold: f64,          // 触发恐惧脉冲的阈值
    
    // 参数
    #[pyo3(get)]
    pub surprise_threshold: f64,      // 触发完整蒸馏的阈值
    #[pyo3(get)]
    pub base_surprise_threshold: f64, // 基础阈值，用于从恐惧脉冲中恢复
    #[pyo3(get)]
    pub decay_rate: f64,              // 预测衰减率
    #[pyo3(get)]
    pub window_size: usize,           // 历史窗口大小
}

#[pymethods]
impl SurpriseFilter {
    #[new]
    #[pyo3(signature = (surprise_threshold=0.4, decay_rate=0.95, window_size=100, vitality_decay_rate=0.05, fear_threshold=0.3))]
    fn new(surprise_threshold: f64, decay_rate: f64, window_size: usize, vitality_decay_rate: f64, fear_threshold: f64) -> Self {
        SurpriseFilter {
            prediction_cache: HashMap::new(),
            running_surprise: 0.0,
            spike_history: VecDeque::new(),
            vitality: 1.0,
            vitality_decay_rate,
            fear_threshold,
            surprise_threshold,
            base_surprise_threshold: surprise_threshold,
            decay_rate,
            window_size,
        }
    }
    
    /// 执行生命周期：随时间衰减生命值，并检查是否触发恐惧脉冲
    fn tick_vitality(&mut self) -> bool {
        self.vitality -= self.vitality_decay_rate;
        if self.vitality < 0.0 {
            self.vitality = 0.0;
        }
        
        // 恐惧脉冲：如果生命值过低，极度调高敏感度（降低惊奇度阈值）
        if self.vitality < self.fear_threshold {
            self.surprise_threshold = self.base_surprise_threshold * 0.2; // 极其敏感，什么都大惊小怪
            true // 触发了恐惧脉冲
        } else {
            self.surprise_threshold = self.base_surprise_threshold;
            false
        }
    }
    
    /// 喂食系统：当系统获得有价值的信息（降低了熵增）时，恢复生命值
    fn feed_vitality(&mut self, amount: f64) {
        self.vitality += amount;
        if self.vitality > 1.0 {
            self.vitality = 1.0;
        }
    }
    
    /// Getter: 获取惊奇度阈值
    #[getter]
    fn get_surprise_threshold(&self) -> f64 {
        self.surprise_threshold
    }
    
    /// Getter: 获取衰减率
    #[getter]
    fn get_decay_rate(&self) -> f64 {
        self.decay_rate
    }

    /// 计算预测误差（惊奇度）
    /// 输入：实际 embedding 和预测 embedding
    /// 输出：惊奇度 (0.0-1.0)
    fn compute_prediction_error(
        &self,
        actual_embedding: Vec<f64>,
        predicted_embedding: Vec<f64>,
    ) -> f64 {
        if actual_embedding.is_empty() || predicted_embedding.is_empty() {
            return 0.5; // 默认中等惊奇度
        }

        // 计算 L2 距离
        let mut sum_sq = 0.0;
        for i in 0..actual_embedding.len().min(predicted_embedding.len()) {
            let diff = actual_embedding[i] - predicted_embedding[i];
            sum_sq += diff * diff;
        }

        let distance = sum_sq.sqrt();
        
        // 用更平缓的函数映射到 (0, 1) 范围，使区分度更高
        // 降低灵敏度，使得大部分普通对话的误差不会过高
        let scale_factor = 2.0;
        1.0 - (-distance / scale_factor).exp()
    }

    /// 评估脉冲：是否超过阈值？
    fn should_fire_spike(&self, surprise_score: f64) -> bool {
        surprise_score >= self.surprise_threshold
    }

    /// 记录脉冲事件，并在脑干光纤中传播
    fn record_spike(&mut self, mut spike: NeuralSpike) {
        // 根据惊奇度自动分配脑干光纤通道
        if spike.surprise_score > 0.8 {
            spike.fiber_channel = "brainstem_emergency_fiber".to_string(); // 极高惊奇，紧急中断
        } else if spike.surprise_score > self.surprise_threshold {
            spike.fiber_channel = "cortex_routing_fiber".to_string(); // 高惊奇，皮层路由
        } else {
            spike.fiber_channel = "autonomic_cache_fiber".to_string(); // 低惊奇，自主神经缓存
        }
        
        self.spike_history.push_back(spike.clone());
        
        // 维持窗口大小
        while self.spike_history.len() > self.window_size {
            self.spike_history.pop_front();
        }
        
        // 更新运行平均
        if !self.spike_history.is_empty() {
            let avg: f64 = self.spike_history
                .iter()
                .map(|s| s.surprise_score)
                .sum::<f64>() / self.spike_history.len() as f64;
            self.running_surprise = avg;
        }
    }

    /// 获取运行窗口内的脉冲统计
    fn get_spike_statistics(&self) -> (f64, usize, f64) {
        // (平均惊奇度, 脉冲计数, 最大惊奇度)
        if self.spike_history.is_empty() {
            return (0.0, 0, 0.0);
        }

        let avg_surprise: f64 = self.spike_history
            .iter()
            .map(|s| s.surprise_score)
            .sum::<f64>() / self.spike_history.len() as f64;

        let max_surprise = self.spike_history
            .iter()
            .map(|s| s.surprise_score)
            .fold(0.0, f64::max);

        (avg_surprise, self.spike_history.len(), max_surprise)
    }

    /// 预测衰减（类似人类遗忘曲线）
    fn decay_predictions(&mut self) {
        for (_, (_, ref mut confidence)) in self.prediction_cache.iter_mut() {
            *confidence *= self.decay_rate;
        }
    }

    /// 清理低置信度的预测
    fn prune_predictions(&mut self, min_confidence: f64) {
        self.prediction_cache.retain(|_, (_, conf)| *conf > min_confidence);
    }
}

// ============================================================================
// 2. LOCAL CONSOLIDATION - 局部巩固
// ============================================================================

/// 碎片化节点 - 需要被合并的弱连接节点
#[derive(Clone, Debug)]
pub struct FragmentedNode {
    pub node_id: String,
    pub semantic_vector: Vec<f64>,
    pub connectivity_score: f64,      // 0.0-1.0: 连接强度
    pub age_days: u32,                // 存在天数
    pub fragment_group: Vec<String>,  // 相关碎片组
}

/// 巩固块 - 合并后的高维语义块
#[pyclass(module = "bdm_rust")]
#[derive(Clone, Debug)]
pub struct ConsolidatedBlock {
    #[pyo3(get)]
    pub consolidation_id: String,
    #[pyo3(get)]
    pub member_nodes: Vec<String>,    // 包含的原始节点
    #[pyo3(get)]
    pub meta_semantic: Vec<f64>,      // 高维元语义向量
    #[pyo3(get)]
    pub consolidation_score: f64,     // 巩固质量 (0.0-1.0)
    #[pyo3(get)]
    pub timestamp: i64,
    #[pyo3(get)]
    pub collective_vitality: f64,     // 集体生命值 (Eros of Consolidation)
}

#[pymethods]
impl ConsolidatedBlock {
    #[new]
    fn new(consolidation_id: String, member_nodes: Vec<String>, meta_semantic: Vec<f64>, collective_vitality: f64) -> Self {
        ConsolidatedBlock {
            consolidation_id,
            member_nodes,
            meta_semantic,
            consolidation_score: 1.0,
            timestamp: 0,
            collective_vitality,
        }
    }
    
    /// Getter: 获取巩固块ID
    #[getter]
    fn get_consolidation_id(&self) -> String {
        self.consolidation_id.clone()
    }
    
    /// Getter: 获取成员节点
    #[getter]
    fn get_member_nodes(&self) -> Vec<String> {
        self.member_nodes.clone()
    }
    
    /// Getter: 获取元语义向量
    #[getter]
    fn get_meta_semantic(&self) -> Vec<f64> {
        self.meta_semantic.clone()
    }
    
    /// Getter: 获取巩固质量分
    #[getter]
    fn get_consolidation_score(&self) -> f64 {
        self.consolidation_score
    }
}

/// 局部巩固引擎 - 后台异步任务
#[pyclass(module = "bdm_rust")]
pub struct LocalConsolidationEngine {
    pub min_fragment_size: usize,      // 触发合并的最小碎片数
    pub max_consolidation_distance: f64, // 最大合并距离
    pub consolidation_history: Vec<ConsolidatedBlock>,
}

#[pymethods]
impl LocalConsolidationEngine {
    #[new]
    #[pyo3(signature = (min_fragment_size=3, max_consolidation_distance=0.5))]
    fn new(min_fragment_size: usize, max_consolidation_distance: f64) -> Self {
        LocalConsolidationEngine {
            min_fragment_size,
            max_consolidation_distance,
            consolidation_history: Vec::new(),
        }
    }

    /// 识别需要巩固的碎片组
    /// 使用聚类算法找到语义相近的节点
    fn identify_fragment_clusters(
        &self,
        nodes: Vec<(String, Vec<f64>)>,
    ) -> Vec<Vec<String>> {
        if nodes.is_empty() {
            return Vec::new();
        }

        let mut clusters = Vec::new();
        let mut assigned = vec![false; nodes.len()];

        // 简单的贪心聚类
        for i in 0..nodes.len() {
            if assigned[i] {
                continue;
            }

            let mut cluster = vec![nodes[i].0.clone()];
            assigned[i] = true;

            // 找到相近的节点
            for j in (i + 1)..nodes.len() {
                if assigned[j] {
                    continue;
                }

                let distance = cosine_distance(&nodes[i].1, &nodes[j].1);
                if distance < self.max_consolidation_distance {
                    cluster.push(nodes[j].0.clone());
                    assigned[j] = true;
                }
            }

            if cluster.len() >= self.min_fragment_size {
                clusters.push(cluster);
            }
        }

        clusters
    }

    /// 生成巩固块 - 合并碎片的元语义，并保留因果链条 (DAG)
    /// 引入 "Eros of Consolidation" - 合并即生存，集体提高 Vitality
    fn consolidate_cluster(
        &mut self,
        cluster: Vec<String>,
        embeddings: Vec<Vec<f64>>,
        base_vitality: f64, // 传入当前的平均生命值
    ) -> ConsolidatedBlock {
        // 计算元语义向量（平均 + 方差）
        let mut meta_semantic = vec![0.0; embeddings[0].len()];
        
        for embedding in &embeddings {
            for (i, val) in embedding.iter().enumerate() {
                meta_semantic[i] += val;
            }
        }
        
        // 平均化
        for val in &mut meta_semantic {
            *val /= embeddings.len() as f64;
        }

        let consolidation_id = format!("consolidated_{}", uuid::Uuid::new_v4());
        
        // Eros of Consolidation: 碎片合并后，生命值获得非线性提升，对抗死亡
        // 比如 3 个碎片合并，生命值不是简单的平均，而是有额外奖励
        let cluster_size = cluster.len() as f64;
        let collective_vitality = (base_vitality * (1.0 + (cluster_size * 0.1))).clamp(0.0, 1.0);
        
        // 这里的 cluster 包含了所有的原节点 ID
        // ConsolidatedBlock 会记录下这些成员节点，从而在 Python 层重建或者保留 DAG 的因果关系
        let block = ConsolidatedBlock::new(consolidation_id.clone(), cluster, meta_semantic, collective_vitality);
        
        self.consolidation_history.push(block.clone());
        block
    }

    /// 获取巩固历史统计
    fn get_consolidation_stats(&self) -> (usize, usize) {
        // (总巩固次数, 涉及总节点数)
        let total_blocks = self.consolidation_history.len();
        let total_nodes: usize = self.consolidation_history
            .iter()
            .map(|b| b.member_nodes.len())
            .sum();
        
        (total_blocks, total_nodes)
    }
}

/// 计算余弦相似度（独立函数，非方法）
fn cosine_distance(v1: &[f64], v2: &[f64]) -> f64 {
    if v1.is_empty() || v2.is_empty() {
        return 1.0;
    }

    let mut dot_product = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;

    let len = v1.len().min(v2.len());
    for i in 0..len {
        dot_product += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }

    let norm1 = norm1.sqrt();
    let norm2 = norm2.sqrt();

    if norm1 == 0.0 || norm2 == 0.0 {
        return 1.0;
    }

    1.0 - (dot_product / (norm1 * norm2))
}

// ============================================================================
// 3. MOE-BASED WORLD MODEL - 多专家系统
// ============================================================================

/// 专家配置
#[derive(Clone, Debug)]
pub struct ExpertSpec {
    pub expert_id: String,
    pub expert_type: String,          // "math", "physics", "logic", "memory"
    pub weight: f64,                  // 0.0-1.0 初始权重
    pub activation_temperature: f64,  // 激活温度
}

/// MoE 路由器 - 决定激活哪个专家
#[pyclass(module = "bdm_rust")]
pub struct MoERouter {
    pub experts: Vec<ExpertSpec>,
    pub routing_cache: HashMap<String, Vec<f64>>, // node_id -> expert_logits
    pub expert_load: HashMap<String, usize>,      // expert_id -> 负载
}

#[pymethods]
impl MoERouter {
    #[new]
    fn new() -> Self {
        MoERouter {
            experts: Vec::new(),
            routing_cache: HashMap::new(),
            expert_load: HashMap::new(),
        }
    }

    /// 注册专家
    fn register_expert(
        &mut self,
        expert_id: String,
        expert_type: String,
        weight: f64,
    ) {
        let spec = ExpertSpec {
            expert_id: expert_id.clone(),
            expert_type,
            weight,
            activation_temperature: 1.0,
        };
        self.experts.push(spec);
        self.expert_load.insert(expert_id, 0);
    }

    /// 根据 DAG 当前活跃分支计算路由权重
    /// 输入：当前节点 ID、DAG 路径
    /// 输出：每个专家的激活概率
    fn route(
        &mut self,
        current_node_id: String,
        dag_context: Vec<String>,
    ) -> Vec<(String, f64)> {
        let mut logits = vec![0.0; self.experts.len()];

        // 基于 DAG 路径和节点内容计算亲和力
        for (i, expert) in self.experts.iter().enumerate() {
            let affinity = match expert.expert_type.as_str() {
                "math" => {
                    // 数学专家：仅对明显的数学计算请求敏感，避免误伤带有数字的故事设定
                    if current_node_id.contains("计算") || current_node_id.contains("等于") || 
                       current_node_id.contains("多少") || current_node_id.contains("算一下") {
                        if current_node_id.contains("=") || current_node_id.contains("+") || 
                           current_node_id.contains("-") || current_node_id.contains("*") || 
                           current_node_id.contains("/") || current_node_id.contains("^") ||
                           current_node_id.chars().any(|c| c.is_digit(10)) {
                            0.9
                        } else {
                            0.3
                        }
                    } else {
                        0.1
                    }
                }
                "physics" => {
                    // 物理专家：对物理名词敏感
                    if current_node_id.contains("光速") || current_node_id.contains("速度") || 
                       current_node_id.contains("时间") || current_node_id.contains("空间") ||
                       current_node_id.contains("引力") || current_node_id.contains("质量") {
                        0.9
                    } else {
                        0.1
                    }
                }
                "logic" => {
                    // 逻辑专家：对推理词汇敏感
                    if current_node_id.contains("为什么") || current_node_id.contains("原因") || 
                       current_node_id.contains("假设") || current_node_id.contains("如果") ||
                       current_node_id.contains("逻辑") || current_node_id.contains("分析") {
                        0.8
                    } else {
                        0.4 // 默认的基础兜底专家
                    }
                }
                "memory" => {
                    // 记忆专家：对历史查询词汇敏感
                    if current_node_id.contains("之前") || current_node_id.contains("记得") || 
                       current_node_id.contains("刚才") || current_node_id.contains("回顾") ||
                       current_node_id.contains("历史") || current_node_id.contains("设定") {
                        0.95
                    } else if !dag_context.is_empty() {
                        0.6 // 有上下文时也倾向于激活记忆专家
                    } else {
                        0.1
                    }
                }
                _ => 0.1,
            };

            // 引入一定的随机性（温度），防止永远固定路由到同一个专家
            let temp = expert.activation_temperature.max(0.1);
            logits[i] = (affinity * expert.weight) / temp;
        }

        // Softmax 归一化
        let max_logit = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f64 = logits.iter().map(|x| (*x - max_logit).exp()).sum();
        
        let mut result: Vec<_> = self.experts
            .iter()
            .zip(logits.iter())
            .map(|(exp, logit)| (exp.expert_id.clone(), (*logit - max_logit).exp() / sum))
            .collect();

        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// 更新专家负载（轮询平衡）
    fn update_expert_load(&mut self, expert_id: String) {
        if let Some(load) = self.expert_load.get_mut(&expert_id) {
            *load += 1;
        }
    }

    /// 获取负载最少的 K 个专家
    fn get_top_k_experts(&self, k: usize) -> Vec<String> {
        let mut experts_by_load: Vec<_> = self.expert_load.iter().collect();
        experts_by_load.sort_by_key(|(_, load)| *load);
        
        experts_by_load
            .iter()
            .take(k)
            .map(|(id, _)| (*id).clone())
            .collect()
    }
}

/// 世界模型执行器
#[pyclass(module = "bdm_rust")]
pub struct WorldModelExecutor {
    pub router: MoERouter,
    pub execution_log: Vec<(String, String, f64)>, // (node_id, expert_id, timestamp)
    pub world_state: HashMap<String, f64>,         // 世界状态变量
    
    // 自然选择与微突变 (Micro-Mutation & Selection)
    pub branch_fitness: HashMap<String, f64>,      // 专家分支 -> 适应度 (0.0 - 1.0)
    pub token_allocation: HashMap<String, usize>,  // 专家分支 -> 分配的 Token 空间
}

#[pymethods]
impl WorldModelExecutor {
    #[new]
    fn new() -> Self {
        let mut executor = WorldModelExecutor {
            router: MoERouter::new(),
            execution_log: Vec::new(),
            world_state: HashMap::new(),
            branch_fitness: HashMap::new(),
            token_allocation: HashMap::new(),
        };
        
        // 初始化每个分支的 Token 空间
        executor.token_allocation.insert("math".to_string(), 256);
        executor.token_allocation.insert("physics".to_string(), 256);
        executor.token_allocation.insert("logic".to_string(), 256);
        executor.token_allocation.insert("memory".to_string(), 256);
        
        // 初始化适应度
        executor.branch_fitness.insert("math".to_string(), 0.5);
        executor.branch_fitness.insert("physics".to_string(), 0.5);
        executor.branch_fitness.insert("logic".to_string(), 0.5);
        executor.branch_fitness.insert("memory".to_string(), 0.5);
        
        executor
    }
    
    /// 评估分支适应度，如果分支表现不佳（未能缓解恐惧脉冲），则进行微小惩罚。
    /// 极低适应度的分支将被直接剪枝（Token 空间释放给其他分支）
    fn evaluate_and_select(&mut self, fear_resolved: bool, active_expert_type: String) -> Vec<String> {
        let mut pruned_branches = Vec::new();
        
        if let Some(fitness) = self.branch_fitness.get_mut(&active_expert_type) {
            if fear_resolved {
                *fitness = (*fitness + 0.05).clamp(0.0, 1.0); // 奖励：缓解了恐惧
            } else {
                *fitness = (*fitness - 0.05).clamp(0.0, 1.0); // 惩罚：未缓解恐惧
            }
            
            // 死亡判定 (Pruning)
            if *fitness <= 0.1 {
                pruned_branches.push(active_expert_type.clone());
            }
        }
        
        // 优胜劣汰：释放死亡分支的 Token 空间，按比例分配给幸存者
        for pruned in &pruned_branches {
            if let Some(freed_tokens) = self.token_allocation.remove(pruned) {
                // 将被剪枝的分支适应度设为 0 (逻辑死亡)
                self.branch_fitness.insert(pruned.clone(), 0.0);
                
                // 将释放的 Token 平均分配给适应度最高的分支
                let mut best_branch = "".to_string();
                let mut max_fit = 0.0;
                for (branch, fit) in &self.branch_fitness {
                    if *fit > max_fit {
                        max_fit = *fit;
                        best_branch = branch.clone();
                    }
                }
                
                if !best_branch.is_empty() {
                    if let Some(alloc) = self.token_allocation.get_mut(&best_branch) {
                        *alloc += freed_tokens;
                    }
                }
            }
        }
        
        // 微突变：对所有存活分支的权重进行 +/- 0.001 级别的微调
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for expert in &mut self.router.experts {
            let mutation = rng.gen_range(-0.001..=0.001);
            expert.weight = (expert.weight + mutation).clamp(0.0, 1.0);
        }
        
        pruned_branches
    }

    /// 模拟一步世界更新
    /// 返回下一步预测状态变量
    fn simulate_step(
        &mut self,
        current_node_id: String,
        dag_context: Vec<String>,
        input_variables: Vec<(String, f64)>,
    ) -> Vec<(String, f64)> {
        // 1. 路由：选择哪个专家激活
        let expert_routing = self.router.route(current_node_id.clone(), dag_context);

        // 2. 执行：选择概率最高的专家
        if !expert_routing.is_empty() {
            let (chosen_expert, _prob) = &expert_routing[0];
            self.router.update_expert_load(chosen_expert.clone());
            
            // 3. 模型推断：根据专家类型计算输出
            let mut output_variables = input_variables.clone();
            
            for (var_name, var_value) in &mut output_variables {
                *var_value = match chosen_expert.as_str() {
                    exp if exp.contains("math") => {
                        // 数学变换：平方根
                        var_value.abs().sqrt()
                    }
                    exp if exp.contains("physics") => {
                        // 物理变换：阻尼
                        *var_value * 0.95
                    }
                    exp if exp.contains("logic") => {
                        // 逻辑变换：阈值
                        if *var_value > 0.5 { 1.0 } else { 0.0 }
                    }
                    _ => *var_value, // 默认保持
                };
            }

            // 4. 记录执行日志
            self.execution_log.push((
                current_node_id,
                chosen_expert.clone(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64(),
            ));

            output_variables
        } else {
            input_variables
        }
    }

    /// 获取执行统计
    fn get_execution_stats(&self) -> (usize, Vec<(String, usize)>) {
        // (总执行数, [(expert_id, 执行计数)])
        let mut expert_counts: HashMap<String, usize> = HashMap::new();
        
        for (_, expert_id, _) in &self.execution_log {
            *expert_counts.entry(expert_id.clone()).or_insert(0) += 1;
        }

        let mut counts_vec: Vec<_> = expert_counts.into_iter().collect();
        counts_vec.sort_by_key(|a| std::cmp::Reverse(a.1));

        (self.execution_log.len(), counts_vec)
    }
}

// 需要在 Cargo.toml 中添加 uuid crate
// 这里使用简化的 UUID 生成
mod simple_uuid {
    use std::sync::atomic::{AtomicU64, Ordering};
    
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    
    pub struct Uuid(String);
    
    impl Uuid {
        pub fn new_v4() -> Uuid {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            Uuid(format!("uuid-{}", n))
        }
    }
    
    impl std::fmt::Display for Uuid {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
}

use simple_uuid as uuid;
