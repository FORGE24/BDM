import random
import copy
from typing import Dict, Any, List, Optional

class SystemState:
    """系统运行时的状态数据载体"""
    def __init__(self):
        self.task_success_rate: float = 0.8
        self.memory_fidelity_score: float = 0.9
        self.average_response_time: float = 1.2
        self.user_engagement_score: float = 0.7
        self.cost_per_conversation: float = 0.05

class FitnessFunction:
    """定义系统的适应度（生存度）"""
    def __init__(self):
        # 根据 config/system.yaml 的默认权重设置
        self.weights = {
            "task_success": 0.4,
            "memory_fidelity": 0.3,
            "user_engagement": 0.2,
            "response_time": -0.05,
            "cost_efficiency": -0.05
        }
    
    def calculate_fitness(self, system_state: SystemState) -> float:
        """计算当前系统适应度"""
        fitness = (
            self.weights["task_success"] * system_state.task_success_rate +
            self.weights["memory_fidelity"] * system_state.memory_fidelity_score +
            self.weights["user_engagement"] * system_state.user_engagement_score +
            self.weights["response_time"] * system_state.average_response_time +
            self.weights["cost_efficiency"] * system_state.cost_per_conversation
        )
        return fitness


class EvolutionaryOptimizer:
    """驱动系统自我优化的进化引擎"""
    def __init__(self):
        self.current_params = self.load_default_params()
        self.generation = 0
        self.fitness_history: List[float] = []
        self.fitness_function = FitnessFunction()
        
    def load_default_params(self) -> Dict[str, float]:
        """加载初始系统参数"""
        return {
            "max_tokens": 1000.0,
            "overlap_tokens": 100.0,
            "base_forgetting_rate": 0.01,
            "recovery_threshold": 0.7,
            "vector_weight": 0.7,
            "keyword_weight": 0.3
        }
        
    def collect_performance_data(self, metrics: Dict[str, float] = None) -> SystemState:
        """收集系统的真实运行表现数据"""
        state = SystemState()
        if metrics:
            state.task_success_rate = metrics.get("task_success_rate", state.task_success_rate)
            state.memory_fidelity_score = metrics.get("memory_fidelity_score", state.memory_fidelity_score)
            state.average_response_time = metrics.get("average_response_time", state.average_response_time)
            state.user_engagement_score = metrics.get("user_engagement_score", state.user_engagement_score)
            state.cost_per_conversation = metrics.get("cost_per_conversation", state.cost_per_conversation)
        return state
        
    def needs_evolution(self, current_fitness: float) -> bool:
        """判断是否需要启动进化"""
        if len(self.fitness_history) < 2:
            return True
        # 如果适应度下降或者长期没有明显提升，则触发进化
        avg_recent = sum(self.fitness_history[-3:]) / min(3, len(self.fitness_history))
        return current_fitness <= avg_recent * 1.05
        
    def mutate_value(self, base_val: float, min_val: float, max_val: float, mutation_strength: float) -> float:
        """对单个参数进行小幅度变异"""
        delta = base_val * mutation_strength * random.uniform(-1, 1)
        new_val = base_val + delta
        # 如果是整数类型的参数（例如 tokens），在这里处理可能产生的浮点，但为了一致性保留为 float
        return max(min_val, min(max_val, new_val))

    def generate_mutations(self, base_params: Dict[str, float]) -> List[Dict[str, float]]:
        """在当前参数基础上生成多个变异候选者"""
        candidates = []
        for _ in range(10):  # 生成10个候选
            candidate = copy.deepcopy(base_params)
            
            candidate["max_tokens"] = self.mutate_value(base_params["max_tokens"], 500, 5000, 0.2)
            candidate["overlap_tokens"] = self.mutate_value(base_params["overlap_tokens"], 10, 500, 0.2)
            candidate["base_forgetting_rate"] = self.mutate_value(base_params["base_forgetting_rate"], 0.001, 0.1, 0.1)
            candidate["recovery_threshold"] = self.mutate_value(base_params["recovery_threshold"], 0.3, 0.9, 0.1)
            candidate["vector_weight"] = self.mutate_value(base_params["vector_weight"], 0.1, 0.9, 0.15)
            
            # 保证权重之和逻辑一致
            candidate["keyword_weight"] = 1.0 - candidate["vector_weight"]
            
            candidates.append(candidate)
        return candidates

    def select_best_candidate(self, candidates: List[Dict[str, float]]) -> Dict[str, float]:
        """评估候选参数并选择最优（模拟 A/B 测试）"""
        # 在真实系统中，需要用各个候选参数运行一小段时间或在验证集上跑
        # 这里用一个简单的启发式来模拟：假设特定方向的参数能提升特定适应度
        best_candidate = candidates[0]
        best_score = -float('inf')
        
        for candidate in candidates:
            # 模拟：适当增加 max_tokens 和调整 vector_weight 会提升适应度
            simulated_score = (
                candidate["max_tokens"] * 0.001 + 
                candidate["vector_weight"] * 2.0 - 
                candidate["base_forgetting_rate"] * 10.0
            )
            if simulated_score > best_score:
                best_score = simulated_score
                best_candidate = candidate
                
        return best_candidate

    def apply_new_parameters(self, new_params: Dict[str, float]):
        """应用新的系统参数"""
        self.current_params = new_params
        
    def evolution_cycle(self, metrics: Dict[str, float] = None) -> tuple[bool, Optional[Dict[str, float]]]:
        """执行一次完整的进化周期"""
        performance_data = self.collect_performance_data(metrics)
        current_fitness = self.fitness_function.calculate_fitness(performance_data)
        self.fitness_history.append(current_fitness)
        
        print(f"  [进化引擎] 当前代数: {self.generation}, 当前适应度: {current_fitness:.4f}")
        
        if self.needs_evolution(current_fitness):
            print(f"  [进化引擎] 满足进化条件，开始繁衍变异...")
            candidate_params = self.generate_mutations(self.current_params)
            best_params = self.select_best_candidate(candidate_params)
            self.apply_new_parameters(best_params)
            self.generation += 1
            return True, best_params
            
        return False, None
