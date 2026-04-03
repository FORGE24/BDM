import os
import time
from dotenv import load_dotenv
import bdm_rust
from memory import MemoryChunk
from distiller import self_distillation, call_llm
from database import DatabaseManager
from evolution import EvolutionaryOptimizer
from safety import SafetyGuardrail
from retriever import MemoryRetriever
from intent import check_task_completion

load_dotenv()

class DialogManager:
    def __init__(self, max_tokens=1000):
        self.evolution_engine = EvolutionaryOptimizer()
        self.safety_guardrail = SafetyGuardrail()
        
        # 从进化引擎获取初始参数
        params = self.evolution_engine.current_params
        
        self.max_tokens = int(params["max_tokens"])
        self.chunking_config = bdm_rust.ChunkingConfig(
            max_tokens=self.max_tokens, 
            overlap_tokens=int(params["overlap_tokens"])
        )
        self.dynamic_forgetting = bdm_rust.DynamicForgetting(
            base_forgetting_rate=params["base_forgetting_rate"], 
            recovery_threshold=params["recovery_threshold"]
        )
        
        self.db_manager = DatabaseManager()
        self.current_stream = [] # 记录当前未分块的对话流
        self.memory_database = self.db_manager.load_memory_chunks()  # 从数据库加载记忆
        self.last_distilled_memory = None  # 记录上一轮的蒸馏记忆
        self.retriever = MemoryRetriever(self.db_manager)
        
        # 用于反馈进化引擎的真实指标收集
        self.session_metrics = {
            "tasks_completed": 0,
            "total_turns": 0,
            "total_response_time": 0.0,
            "total_tokens_used": 0,
            "fidelity_scores": []
        }
        
    def generate_response(self, user_input: str) -> str:
        """生成大模型响应，结合相关历史记忆"""
        
        # 1. 检索历史记忆
        context = self.retriever.retrieve_context(user_input)
        
        # 2. 如果存在上一轮的级联记忆，也放入上下文中，以保证对话极强的连贯性
        recent_context = ""
        if self.last_distilled_memory:
            recent_context = f"""
【刚刚发生的对话精华】
- 实体: {self.last_distilled_memory.entities}
- 关键决策: {self.last_distilled_memory.decisions}
- 事实与信息: {self.last_distilled_memory.important_facts}
            """

        system_msg = f"""
你是一个基于 MLED (机器生命进化蒸馏系统) 的 AI。
你拥有记忆能力，以下是从你的长期记忆数据库中检索到的可能相关的背景信息：

【历史记忆库】
{context}
{recent_context}

请利用上述背景知识（如果相关的话）以及自身的知识储备，自然地回复用户。
如果你觉得历史记忆不相关，请忽略它。
"""
        # 调用大模型生成文本，不需要 json 格式
        reply = call_llm(
            prompt=user_input, 
            system_msg=system_msg,
            response_format={"type": "text"}
        )
        return reply
        
    def process_utterance(self, user_input: str):
        """处理用户的每一次输入"""
        
        # 1. 触发安全护栏检测
        safety_check = self.safety_guardrail.enforce_safety(user_input)
        if not safety_check.passed:
            print(safety_check.response)
            return
            
        # 2. 实时生成回应 (带有记忆检索)
        start_time = time.time()
        reply = self.generate_response(user_input)
        elapsed_time = time.time() - start_time
        
        self.session_metrics["total_turns"] += 1
        self.session_metrics["total_response_time"] += elapsed_time
        self.session_metrics["total_tokens_used"] += bdm_rust.count_tokens(user_input + reply)
        
        print(f"🤖 MLED: {reply}")
        
        # 3. 意图识别检查：如果用户发言代表了话题的自然结束，则主动截断流
        task_complete = check_task_completion(user_input)
        if task_complete:
            self.session_metrics["tasks_completed"] += 1
        
        # 将用户的输入和系统回复一并加入当前流
        combined_interaction = f"用户: {user_input}\n系统: {reply}"
        self.current_stream.append(combined_interaction)
        
        if task_complete:
            print("[系统] 检测到话题结束/任务完成，主动触发记忆截断封装...")
            self._flush_current_chunk()
            return
            
        # 4. 调用 Rust 层的智能分块算法
        chunks = bdm_rust.intelligent_chunking(self.current_stream, self.chunking_config)
        
        # intelligent_chunking 返回所有划分好的块。如果返回了多个块，
        # 意味着前面的对话已经形成了一个完整的 MemoryChunk（甚至多个）。
        # 最后一个块往往是不完整的（或者是恰好完整），在我们的实时对话流中，
        # 我们把形成完整边界的块拿出来进行蒸馏和存储，剩余的内容留作下一步的流。
        if len(chunks) > 1:
            # 取出已经确认边界的所有前面的块
            completed_chunks = chunks[:-1]
            for chunk in completed_chunks:
                self._process_completed_chunk(chunk)
                
            # 将 current_stream 更新为最后一个未完成的块所包含的内容
            # (在实际复杂系统中，这里可能需要提取最后一个块的 raw_text 重新打散，
            # 这里为简化实现，只保留最后一条输入或清空)
            last_chunk = chunks[-1]
            self.current_stream = last_chunk.raw_text.split("\n")
        else:
            # 只有1个块，说明还没达到阈值或者语义边界，继续积累
            pass
            
        print(f"[系统] 已记录您的输入 (当前流积累 Token数估计: {bdm_rust.count_tokens(user_input)})")
        
    def _process_completed_chunk(self, chunk: MemoryChunk):
        """处理已经形成边界的完整记忆块"""
        print(f"\n[系统] 智能分块触发！形成新记忆块 (Token数: {chunk.tokens}, 语义边界: {chunk.semantic_boundary})")
        print(f"[系统] 正在触发级联自蒸馏机制... (将融合上一轮 50% 的记忆精华)")
        
        # 调用基于 Deepseek API 的自蒸馏引擎，并传入上一轮的记忆
        distilled = self_distillation(chunk, previous_distilled=self.last_distilled_memory, compression_level="balanced")
        chunk.distilled_version = distilled
        
        self.memory_database.append(chunk)
        # 更新记录，以便下一个块使用
        self.last_distilled_memory = distilled
        self.session_metrics["fidelity_scores"].append(distilled.fidelity_score)
        
        # 将记忆存入 SQLite 数据库
        if self.db_manager.save_memory_chunk(chunk):
            print("[系统] 记忆已持久化至 SQLite 数据库 (mled_memory.db)")
        else:
            print("[系统] ⚠️ 记忆持久化失败！")
        
        print("====== 蒸馏结果 ======")
        print(f"提取实体: {distilled.entities}")
        print(f"关键决策: {distilled.decisions}")
        print(f"事实与信息: {distilled.important_facts}")
        print("======================\n")
        
    def _flush_current_chunk(self):
        """强制将当前所有流作为完整块处理"""
        if not self.current_stream:
            return
            
        # 强制将整个流合成一个块
        raw_text = "\n".join(self.current_stream)
        tokens = bdm_rust.count_tokens(raw_text)
        chunk = MemoryChunk(raw_text=raw_text, tokens=tokens, semantic_boundary=True)
        
        self._process_completed_chunk(chunk)
        self.current_stream = []

    def run_forgetting_cycle(self):
        """模拟后台运行的遗忘与状态更新周期"""
        print("\n[系统后台] 开始执行动态遗忘与状态更新周期...")
        for i, chunk in enumerate(self.memory_database):
            old_status = chunk.status
            # 调用 Rust 实现的高性能状态更新算法
            updated_chunk = self.dynamic_forgetting.update_memory_state(chunk)
            self.memory_database[i] = updated_chunk
            
            if old_status != updated_chunk.status:
                print(f"  -> 记忆块 [{updated_chunk.chunk_id[:8]}] 状态变更: {old_status} -> {updated_chunk.status}")
        print("[系统后台] 周期执行完毕。\n")

    def run_evolution_cycle(self):
        """执行一次自我进化周期，更新系统的配置参数"""
        print("\n[系统后台] 开始执行自进化周期 (Self-Evolution)...")
        
        # 计算真实收集的指标
        metrics = {}
        if self.session_metrics["total_turns"] > 0:
            metrics["task_success_rate"] = 1.0 if self.session_metrics["tasks_completed"] > 0 else 0.5
            metrics["average_response_time"] = self.session_metrics["total_response_time"] / self.session_metrics["total_turns"]
            metrics["user_engagement_score"] = min(1.0, self.session_metrics["total_turns"] / 10.0) # 假设10轮对话为高参与度
            metrics["cost_per_conversation"] = self.session_metrics["total_tokens_used"] / 10000.0 # 假设10k tokens成本为1
            
        if self.session_metrics["fidelity_scores"]:
            metrics["memory_fidelity_score"] = sum(self.session_metrics["fidelity_scores"]) / len(self.session_metrics["fidelity_scores"])
            
        evolved, new_params = self.evolution_engine.evolution_cycle(metrics)
        
        # 重置部分短期指标（视系统设计而定，这里可以选择重置或保留）
        self.session_metrics["total_turns"] = 0
        self.session_metrics["total_response_time"] = 0.0
        self.session_metrics["total_tokens_used"] = 0
        self.session_metrics["tasks_completed"] = 0
        self.session_metrics["fidelity_scores"] = []
        
        if evolved and new_params:
            print(f"  [进化引擎] 繁衍成功！应用第 {self.evolution_engine.generation} 代新参数:")
            print(f"    - max_tokens: {int(new_params['max_tokens'])}")
            print(f"    - base_forgetting_rate: {new_params['base_forgetting_rate']:.4f}")
            print(f"    - recovery_threshold: {new_params['recovery_threshold']:.4f}")
            print(f"    - vector_weight: {new_params['vector_weight']:.2f}")
            
            # 将新参数应用到 Rust 底层实例中
            self.max_tokens = int(new_params["max_tokens"])
            self.chunking_config = bdm_rust.ChunkingConfig(
                max_tokens=self.max_tokens, 
                overlap_tokens=int(new_params["overlap_tokens"])
            )
            self.dynamic_forgetting = bdm_rust.DynamicForgetting(
                base_forgetting_rate=new_params["base_forgetting_rate"], 
                recovery_threshold=new_params["recovery_threshold"]
            )
        else:
            print("  [进化引擎] 评估后未发现明显改善空间，保持现有参数。")
        print("[系统后台] 自进化周期完毕。\n")

def main():
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("请在 .env 文件中设置 DEEPSEEK_API_KEY")
        return
        
    manager = DialogManager(max_tokens=50) # 设置为较小的阈值以便快速测试分块
    
    print("====== 欢迎来到 MLED 机器生命进化蒸馏系统 ======")
    print("系统已启动，随时准备对话。")
    print("- 输入 'quit' 退出")
    print("- 输入 'flush' 强制记忆蒸馏")
    print("- 输入 'forget' 模拟执行一次动态遗忘周期")
    print("- 输入 'evolve' 模拟执行一次自我进化周期\n")
    
    while True:
        try:
            user_input = input("你: ")
            if user_input.lower() == 'quit':
                break
            if user_input.lower() == 'flush':
                manager._flush_current_chunk()
                continue
            if user_input.lower() == 'forget':
                manager.run_forgetting_cycle()
                continue
            if user_input.lower() == 'evolve':
                manager.run_evolution_cycle()
                continue
                
            if not user_input.strip():
                continue
                
            manager.process_utterance(user_input)
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
