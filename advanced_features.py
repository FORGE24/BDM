"""
高级BDM功能集成层
包括：预测编码、局部巩固、MoE世界模型
"""

import json
import asyncio
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
from bdm_rust import (
    SurpriseFilter, 
    NeuralSpike,
    LocalConsolidationEngine,
    ConsolidatedBlock,
    MoERouter,
    WorldModelExecutor
)


# ============================================================================
# 1. 预测编码与稀疏脉冲系统 (Predictive Coding & Sparse Impulse)
# ============================================================================

class PredictiveCodecInterface:
    """
    预测编码系统的Python接口
    """
    
    def __init__(self, surprise_threshold: float = 0.4):
        """
        初始化预测编码系统
        
        Args:
            surprise_threshold: 触发完整蒸馏的惊奇度阈值(0.0-1.0)
        """
        self.filter = SurpriseFilter(
            surprise_threshold=surprise_threshold,
            decay_rate=0.95,
            window_size=100,
            vitality_decay_rate=0.05,
            fear_threshold=0.3
        )
        self.spike_buffer = []
        self.prediction_cache = {}
        
    def tick(self) -> bool:
        """
        执行一个时间步，衰减生命值
        Returns: 是否触发了恐惧脉冲
        """
        return self.filter.tick_vitality()
        
    def feed_vitality(self, amount: float):
        """喂食生命值（当对话成功降低熵增时调用）"""
        self.filter.feed_vitality(amount)
        
    def compute_surprise(
        self,
        node_id: str,
        actual_embedding: List[float],
        expected_embedding: Optional[List[float]] = None
    ) -> Tuple[float, bool]:
        """
        计算单个节点的"惊奇度"
        
        Args:
            node_id: 节点ID
            actual_embedding: 实际嵌入向量
            expected_embedding: 预期/预测嵌入向量
            
        Returns:
            (惊奇度, 是否应触发脉冲)
        """
        if expected_embedding is None:
            # 如果没有预期值，生成默认预测
            expected_embedding = np.zeros(len(actual_embedding)).tolist()
        
        # 计算预测误差
        surprise_score = self.filter.compute_prediction_error(
            actual_embedding, 
            expected_embedding
        )
        
        # 判断是否应该触发脉冲
        should_fire = self.filter.should_fire_spike(surprise_score)
        
        # 根据惊奇度分配光纤通道（模拟物理脉冲）
        fiber_channel = "autonomic_cache_fiber"
        if surprise_score > 0.8:
            fiber_channel = "brainstem_emergency_fiber"
        elif should_fire:
            fiber_channel = "cortex_routing_fiber"
            
        # 记录脉冲
        spike = NeuralSpike(
            node_id=node_id,
            surprise_score=surprise_score,
            spike_magnitude=surprise_score,
            timestamp=int(datetime.now().timestamp()),
            activation_type="novel" if surprise_score > 0.7 else "routine",
            fiber_channel=fiber_channel
        )
        self.filter.record_spike(spike)
        self.spike_buffer.append(spike)
        
        return surprise_score, should_fire
    
    def get_spike_statistics(self) -> Dict:
        """获取脉冲窗口统计"""
        avg_surprise, spike_count, max_surprise = self.filter.get_spike_statistics()
        return {
            "average_surprise": avg_surprise,
            "spike_count": spike_count,
            "max_surprise": max_surprise,
            "threshold": self.filter.surprise_threshold,
            "vitality": self.filter.vitality, # 新增生命值监控
            "buffer_size": len(self.spike_buffer)
        }
    
    def should_perform_full_consolidation(self) -> bool:
        """
        判断是否应该执行完整的内存蒸馏
        基于运行窗口的平均惊奇度
        """
        avg_surprise, _, _ = self.filter.get_spike_statistics()
        return avg_surprise > self.filter.surprise_threshold
    
    def decay_predictions(self):
        """衰减所有预测（模拟遗忘）"""
        self.filter.decay_predictions()
        self.filter.prune_predictions(min_confidence=0.1)
    
    def get_high_surprise_events(self, top_k: int = 5) -> List[Dict]:
        """获取最高惊奇度的事件"""
        if not self.spike_buffer:
            return []
        
        sorted_spikes = sorted(
            self.spike_buffer,
            key=lambda s: s.surprise_score,
            reverse=True
        )[:top_k]
        
        return [
            {
                "node_id": spike.node_id,
                "surprise_score": spike.surprise_score,
                "timestamp": spike.timestamp,
                "activation_type": spike.activation_type
            }
            for spike in sorted_spikes
        ]


# ============================================================================
# 2. 局部巩固系统 (Local Consolidation)
# ============================================================================

class MemoryConsolidationEngine:
    """
    类似睡眠中的记忆巩固系统
    """
    
    def __init__(self, min_fragment_size: int = 3):
        """
        初始化巩固引擎
        
        Args:
            min_fragment_size: 触发合并的最小碎片数
        """
        self.consolidator = LocalConsolidationEngine(
            min_fragment_size=min_fragment_size,
            max_consolidation_distance=0.5
        )
        self.consolidation_log = []
    
    def identify_fragments(
        self,
        memory_nodes: List[Tuple[str, List[float]]]
    ) -> List[List[str]]:
        """
        识别需要巩固的碎片组
        
        Args:
            memory_nodes: [(node_id, embedding), ...]
            
        Returns:
            碎片组列表
        """
        clusters = self.consolidator.identify_fragment_clusters(memory_nodes)
        return clusters
    
    def consolidate(
        self,
        cluster: List[str],
        embeddings: List[List[float]],
        base_vitality: float = 1.0 # 引入 Eros of Consolidation
    ) -> Dict:
        """
        执行单个碎片组的巩固
        """
        consolidated_block = self.consolidator.consolidate_cluster(
            cluster, embeddings, base_vitality
        )
        
        block_info = {
            "consolidation_id": consolidated_block.consolidation_id,
            "member_nodes": consolidated_block.member_nodes,
            "consolidation_score": consolidated_block.consolidation_score,
            "timestamp": consolidated_block.timestamp,
            "meta_semantic_dimension": len(consolidated_block.meta_semantic),
            "meta_semantic": consolidated_block.meta_semantic,
            "collective_vitality": consolidated_block.collective_vitality # 新增
        }
        
        self.consolidation_log.append(block_info)
        return block_info
    
    def get_consolidation_stats(self) -> Dict:
        """获取巩固统计"""
        total_blocks, total_nodes = self.consolidator.get_consolidation_stats()
        return {
            "total_consolidations": total_blocks,
            "total_consolidated_nodes": total_nodes,
            "average_cluster_size": (
                total_nodes / total_blocks if total_blocks > 0 else 0
            ),
            "consolidation_log_size": len(self.consolidation_log)
        }
    
    async def run_background_consolidation(
        self,
        memory_database,
        interval_seconds: int = 60
    ):
        """
        后台异步运行巩固任务（模拟睡眠）
        
        Args:
            memory_database: 内存数据库连接
            interval_seconds: 巩固周期（秒）
        """
        while True:
            try:
                # 从数据库获取所有节点
                nodes = memory_database.get_all_nodes_with_embeddings()
                
                if nodes:
                    # 识别碎片
                    clusters = self.identify_fragments(nodes)
                    
                    # 执行巩固
                    for cluster, embeddings in clusters:
                        self.consolidate(cluster, embeddings)
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                print(f"巩固任务错误: {e}")
                await asyncio.sleep(interval_seconds)


# ============================================================================
# 3. MoE世界模型 (Multi-Expert Orchestration)
# ============================================================================

class ExpertWorldModel:
    """
    多专家世界模型 - 根据DAG路径激活不同专家
    """
    
    def __init__(self):
        """初始化世界模型"""
        self.router = MoERouter()
        self.executor = WorldModelExecutor()
        self.expert_definitions = {}
        self._setup_default_experts()
    
    def _setup_default_experts(self):
        """设置默认专家"""
        experts = [
            ("math_expert", "math", 0.8),
            ("physics_expert", "physics", 0.7),
            ("logic_expert", "logic", 0.9),
            ("memory_expert", "memory", 0.85),
        ]
        
        for expert_id, expert_type, weight in experts:
            self.router.register_expert(expert_id, expert_type, weight)
            self.expert_definitions[expert_id] = {
                "type": expert_type,
                "weight": weight,
                "description": f"{expert_type.title()} Expert"
            }
    
    def register_expert(
        self,
        expert_id: str,
        expert_type: str,
        weight: float,
        description: str = ""
    ):
        """
        注册新专家
        
        Args:
            expert_id: 专家ID
            expert_type: 专家类型
            weight: 权重(0.0-1.0)
            description: 专家描述
        """
        self.router.register_expert(expert_id, expert_type, weight)
        self.expert_definitions[expert_id] = {
            "type": expert_type,
            "weight": weight,
            "description": description
        }
    
    def route_to_experts(
        self,
        current_node_id: str,
        dag_context: List[str],
        k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        根据当前节点和DAG上下文路由到专家
        
        Args:
            current_node_id: 当前节点ID
            dag_context: DAG路径上下文
            k: 返回顶部K个专家
            
        Returns:
            [(expert_id, probability), ...]
        """
        routing = self.router.route(current_node_id, dag_context)
        return routing[:k]
    
    def execute_world_step(
        self,
        current_node_id: str,
        dag_context: List[str],
        input_variables: Dict[str, float]
    ) -> Dict[str, float]:
        """
        执行一步世界模型更新
        
        Args:
            current_node_id: 当前节点ID
            dag_context: DAG路径
            input_variables: 输入变量字典
            
        Returns:
            输出变量字典
        """
        # 转换为 (name, value) 列表
        input_vars_list = list(input_variables.items())
        
        # 执行模拟
        output_vars_list = self.executor.simulate_step(
            current_node_id,
            dag_context,
            input_vars_list
        )
        
        # 转换回字典
        return dict(output_vars_list)
    
    def get_expert_load_balance(self) -> Dict:
        """获取专家负载均衡情况"""
        top_experts = self.router.get_top_k_experts(len(self.expert_definitions))
        
        return {
            "expert_load": {
                expert_id: self.router.expert_load.get(expert_id, 0)
                for expert_id in self.expert_definitions.keys()
            },
            "least_loaded_experts": top_experts
        }
    
    def get_execution_stats(self) -> Dict:
        """获取执行统计"""
        total_executions, expert_counts = self.executor.get_execution_stats()
        
        return {
            "total_executions": total_executions,
            "expert_invoke_counts": dict(expert_counts),
            "num_registered_experts": len(self.expert_definitions),
            "expertise": self.expert_definitions
        }


# ============================================================================
# 集成接口
# ============================================================================

class AdvancedBDMSystem:
    """
    集成所有三个高级功能的BDM系统
    """
    
    def __init__(self, database_manager, memory_manager):
        """
        初始化高级BDM系统
        
        Args:
            database_manager: 数据库管理器
            memory_manager: 内存管理器
        """
        self.db = database_manager
        self.memory = memory_manager
        
        # 三个主要系统
        self.predictive_codec = PredictiveCodecInterface(surprise_threshold=0.4)
        self.consolidation_engine = MemoryConsolidationEngine()
        self.world_model = ExpertWorldModel()
        
        # 统计信息
        self.stats = {
            "predictive_coding": {},
            "consolidation": {},
            "world_model": {}
        }
    
    def process_node_with_prediction(
        self,
        node_id: str,
        actual_embedding: List[float],
        expected_embedding: Optional[List[float]] = None
    ) -> Dict:
        """
        使用预测编码处理节点
        
        Returns:
            处理结果字典
        """
        surprise, should_fire = self.predictive_codec.compute_surprise(
            node_id, actual_embedding, expected_embedding
        )
        
        return {
            "node_id": node_id,
            "surprise_score": surprise,
            "should_trigger_consolidation": should_fire,
            "spike_stats": self.predictive_codec.get_spike_statistics()
        }
    
    def trigger_consolidation(
        self,
        memory_nodes: List[Tuple[str, List[float]]]
    ) -> List[Dict]:
        """
        触发内存巩固流程
        """
        clusters = self.consolidation_engine.identify_fragments(memory_nodes)
        consolidated_blocks = []
        
        for cluster in clusters:
            # 获取嵌入向量
            embeddings = [node[1] for node in memory_nodes if node[0] in cluster]
            cluster_ids = [node[0] for node in memory_nodes if node[0] in cluster]
            
            if len(embeddings) >= 2:
                block_info = self.consolidation_engine.consolidate(
                    cluster_ids, embeddings
                )
                consolidated_blocks.append(block_info)
        
        return consolidated_blocks
    
    def route_query_to_experts(
        self,
        query_node_id: str,
        dag_context: List[str],
        top_k: int = 3
    ) -> List[Dict]:
        """
        将查询路由到合适的专家
        """
        routing = self.world_model.route_to_experts(
            query_node_id, dag_context, top_k
        )
        
        detailed_routing = []
        for expert_id, probability in routing:
            expert_info = self.world_model.expert_definitions.get(expert_id, {})
            detailed_routing.append({
                "expert_id": expert_id,
                "probability": probability,
                "type": expert_info.get("type", "unknown"),
                "description": expert_info.get("description", "")
            })
        
        return detailed_routing
    
    def get_system_status(self) -> Dict:
        """获取整个系统的状态"""
        return {
            "predictive_coding": self.predictive_codec.get_spike_statistics(),
            "consolidation": self.consolidation_engine.get_consolidation_stats(),
            "world_model": self.world_model.get_execution_stats()
        }
