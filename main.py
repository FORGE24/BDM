import os
import time
import asyncio
import threading
from dotenv import load_dotenv
import bdm_rust
from memory import MemoryChunk
from distiller import self_distillation, call_llm, embedding_model
from database import DatabaseManager
from evolution import EvolutionaryOptimizer
from safety import SafetyGuardrail
from retriever import MemoryRetriever
from intent import check_task_completion
from advanced_features import PredictiveCodecInterface, MemoryConsolidationEngine, ExpertWorldModel, AdvancedBDMSystem

load_dotenv()

class DialogManager:
    def __init__(self, max_tokens=1000):
        self.evolution_engine = EvolutionaryOptimizer()
        self.safety_guardrail = SafetyGuardrail()
        
        # Phase 3: 初始化高级功能
        self.predictive_codec = PredictiveCodecInterface(surprise_threshold=0.3) # 进一步调低阈值，让其更容易跳出缓存，增加活泼度
        self.consolidation_engine = MemoryConsolidationEngine()
        self.world_model = ExpertWorldModel()
        
        # 缓存上一轮预测，用于惊奇度计算
        self.last_expected_embedding = None
        self.consecutive_cache_hits = 0  # 记录连续缓存命中的次数
        self.fear_pulse_active = False   # 记录当前是否处于恐惧脉冲状态
        
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
        
        # 启动后台巩固线程
        self.consolidation_thread = threading.Thread(target=self._start_background_consolidation, daemon=True)
        self.consolidation_thread.start()
        
    def _start_background_consolidation(self):
        """在后台运行巩固任务的包装方法"""
        asyncio.run(self._run_consolidation_loop())
        
    async def _run_consolidation_loop(self):
        """后台异步运行巩固任务（模拟睡眠）"""
        interval_seconds = 60 # 可以配置
        print(f"[系统后台] 记忆巩固引擎已启动，周期为 {interval_seconds} 秒")
        while True:
            try:
                # 只在有一定量新记忆时才执行巩固
                if len(self.current_stream) > 0 or self.session_metrics["total_turns"] < 3:
                    await asyncio.sleep(interval_seconds)
                    continue
                    
                # 从数据库获取所有节点
                nodes = self.db_manager.get_all_nodes_with_embeddings()
                
                # 需要至少有多个节点才值得合并
                if nodes and len(nodes) >= 3:
                    # 识别碎片
                    clusters = self.consolidation_engine.identify_fragments(nodes)
                    
                    # 过滤掉单节点集群
                    valid_clusters = [c for c in clusters if len(c) > 1]
                    
                    if valid_clusters:
                        print(f"\n[系统后台] 触发局部巩固: 发现 {len(valid_clusters)} 个语义碎片组，开始合并...")
                        
                        # 执行巩固
                        for cluster in valid_clusters:
                            embeddings = [n[1] for n in nodes if n[0] in cluster]
                            block_info = self.consolidation_engine.consolidate(cluster, embeddings)
                            
                            # 保存到数据库
                            if self.db_manager.save_consolidated_block(block_info):
                                print(f"  -> [巩固完成] 合并 {len(cluster)} 个节点 -> Super Node [{block_info['consolidation_id'][:8]}]")
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                print(f"巩固任务错误: {e}")
                await asyncio.sleep(interval_seconds)
        
    def generate_response(self, user_input: str, active_expert: str = None) -> str:
        """生成大模型响应，结合相关历史记忆和路由的专家系统提示"""
        
        # 1. 检索历史记忆，并传入当前的 expert 以进行 context_tag 隔离
        context = self.retriever.retrieve_context(user_input, active_expert=active_expert)
        
        # 2. 如果存在上一轮的级联记忆，也放入上下文中，以保证对话极强的连贯性
        recent_context = ""
        if self.last_distilled_memory:
            recent_context = f"""
【刚刚发生的对话精华】
- 实体: {self.last_distilled_memory.entities}
- 关键决策: {self.last_distilled_memory.decisions}
- 事实与信息: {self.last_distilled_memory.important_facts}
            """

        # 3. Phase 3: MoE 专家路由映射到 LLM System Prompt
        expert_prompt = ""
        if active_expert:
            if "logic" in active_expert:
                expert_prompt = "\n【当前激活专家：逻辑推理专家】请你对用户的问题进行严谨的逻辑分析，分步骤推导结论，指出其中的假设和潜在的矛盾。"
            elif "math" in active_expert:
                expert_prompt = "\n【当前激活专家：数学计算专家】请你关注问题中的数值和计算逻辑，确保数学推导的绝对准确性。如果遇到计算请求，请一步步展示计算过程。"
            elif "physics" in active_expert:
                expert_prompt = "\n【当前激活专家：物理科学专家】请你运用物理定律和科学原理来解释用户的问题，注重因果关系和客观规律。"
            elif "memory" in active_expert:
                expert_prompt = "\n【当前激活专家：记忆挖掘专家】请你深度依赖提供的【历史记忆库】，优先从我们过去的对话中寻找答案的线索，建立长期的连贯性。"

        system_msg = f"""
你是一个基于 MLED (机器生命进化蒸馏系统) 的 AI。
你拥有记忆能力，以下是从你的长期记忆数据库中检索到的可能相关的背景信息：

【历史记忆库】
{context}
{recent_context}
{expert_prompt}

请利用上述背景知识（如果相关的话）以及自身的知识储备，自然地回复用户。
如果你觉得历史记忆不相关，请忽略它。
"""
        # 调用大模型生成文本，不需要 json 格式
        reply = call_llm(
            prompt=user_input, 
            system_msg=system_msg
        )
        return reply
        
    def process_utterance(self, user_input: str):
        """处理用户的每一次输入"""
        
        # 1. 触发安全护栏检测
        safety_check = self.safety_guardrail.enforce_safety(user_input)
        if not safety_check.passed:
            print(safety_check.response)
            return
            
        # [新增]: 生命周期滴答 - Vitality 衰减
        self.fear_pulse_active = self.predictive_codec.tick()
        if self.fear_pulse_active:
            print(f"⚠️ [系统警告] 生命值 (Vitality) 极低，触发恐惧脉冲！系统敏感度已调至最高！")
            
        # 2. Phase 3: 预测编码 (Predictive Coding)
        # 生成当前输入的 Embedding
        current_embedding = embedding_model.encode(user_input).tolist()
        
        # 计算惊奇度 (Surprise Score)
        surprise_score, should_fire = self.predictive_codec.compute_surprise(
            node_id=f"input_{self.session_metrics['total_turns']}",
            actual_embedding=current_embedding,
            expected_embedding=self.last_expected_embedding
        )
        
        print(f"[预测编码] 惊奇度: {surprise_score:.3f} | 是否触发深度路由: {should_fire}")
        
        # 如果连续多次命中缓存，强制触发深度路由以保持对话的活力和自然度
        if not should_fire and self.consecutive_cache_hits >= 2:
            print("[预测编码] 连续缓存命中次数达到上限，强制触发深度路由更新预期")
            should_fire = True
            
        start_time = time.time()
        
        if should_fire:
            self.consecutive_cache_hits = 0
            # 惊奇度高，触发完整的大模型路由或生成
            print("[系统] 高惊奇度，触发 MoE 世界模型深度思考...")
            
            # 使用检索器获取 DAG 路径
            context_nodes = self.retriever.retrieve_context(user_input, return_nodes=True) if hasattr(self.retriever, 'retrieve_context') else []
            dag_context = [node.chunk_id for node in context_nodes] if context_nodes else []
            
            # 路由到专家
            routing = self.world_model.route_to_experts(
                current_node_id=f"query_{self.session_metrics['total_turns']}",
                dag_context=dag_context,
                k=1
            )
            
            if routing:
                top_expert, prob = routing[0]
                print(f"[MoE 路由] 激活专家: {top_expert} (亲和力: {prob:.2f})")
                
                # 评估专家适应度并进行自然选择微突变
                # 如果当前处于恐惧脉冲状态，系统急需降低惊奇度，我们假设如果专家被选中，它成功“救场”
                pruned = self.world_model.executor.evaluate_and_select(fear_resolved=self.fear_pulse_active, active_expert_type=top_expert)
                if pruned:
                    print(f"💀 [自然选择] 分支 {pruned} 在恐惧中表现不佳，已被彻底剪枝并触发遗传重组！")
                    
                # 既然专家已经接管并深度思考，系统恢复生命值
                if self.fear_pulse_active:
                    print(f"💚 [生机恢复] {top_expert} 的介入缓解了系统的熵增，生命值恢复。")
                    self.predictive_codec.feed_vitality(0.3)
                    self.fear_pulse_active = False
            else:
                top_expert = None
            
            # 实时生成回应 (带有记忆检索)
            reply = self.generate_response(user_input, active_expert=top_expert)
            
            # 使用反馈信号 (Feedback Signal)：根据恐惧缓解情况，调整刚才使用的历史记忆的 fitness
            if top_expert and self.last_distilled_memory:
                # 简单反馈逻辑：如果没能解决恐惧脉冲（惩罚），或者解决了（奖励）
                # 这里我们假设 retrieval_context 里最近的一条记忆影响了本次决策，因此给予奖励/惩罚
                if self.fear_pulse_active: # 如果依然恐惧，说明刚才检索的记忆没用好
                    self.db_manager.adjust_memory_fitness(self.last_distilled_memory.memory_id, -0.1)
                else:
                    self.db_manager.adjust_memory_fitness(self.last_distilled_memory.memory_id, +0.1)
            
            # 扣除环境资源池能量
            if top_expert:
                tokens_used = bdm_rust.count_tokens(user_input + reply)
                is_alive = self.world_model.executor.consume_energy(top_expert, tokens_used)
                if not is_alive:
                    print(f"⚡ [环境资源枯竭] 专家 {top_expert} 耗尽 Token 能量死亡！已触发 80/20 遗传重组...")
            
            # 暂用当前输入作为下一轮的基准预测
            self.last_expected_embedding = current_embedding
        else:
            self.consecutive_cache_hits += 1
            # 惊奇度低，使用快速缓存或简单规则回复 (节省 Token)
            print(f"[系统] 低惊奇度，使用直觉/缓存直接响应 (连续第 {self.consecutive_cache_hits} 次，节省 80% Token)")
            
            # 使用 LLM 生成轻量级响应，但不进行深度检索和专家路由，降低成本
            # 这里调用 LLM 生成符合上下文的短回复，而不是硬编码的“我明白了”
            system_msg_lite = "你是一个聪明的对话AI。用户说的话在你的意料之中，请简短地（不超过10个字）回应、附和或表示理解，保持对话流畅。"
            reply = call_llm(prompt=user_input, system_msg=system_msg_lite)
            
            self.last_expected_embedding = current_embedding

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
            return reply
            
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
        return reply
        
    def _process_completed_chunk(self, chunk: MemoryChunk):
        """处理已经形成边界的完整记忆块"""
        print(f"\n[系统] 智能分块触发！形成新记忆块 (Token数: {chunk.tokens}, 语义边界: {chunk.semantic_boundary})")
        print(f"[系统] 正在触发级联自蒸馏机制... (已阻断历史污染，执行独立提取)")
        
        # 【新增】构建所有已有记忆块的 ID 列表
        all_memory_ids = [c.chunk_id for c in self.memory_database if hasattr(c, 'chunk_id')]
        
        # 调用基于 Deepseek API 的自蒸馏引擎
        # 在这里我们提取父节点的上下文，作为“血统因子”传给蒸馏器
        parent_contexts = ""
        if hasattr(self.retriever, 'retrieve_context') and self.last_distilled_memory:
            parent_contexts = self.retriever.retrieve_context(chunk.raw_text, max_tokens=300)
            
        distilled = self_distillation(
            chunk, 
            all_memory_ids=all_memory_ids,  # 用于因果链接识别
            parent_contexts=parent_contexts # 引入血统因子，防止错误拦截
        )
        
        # 🚨 淘汰机制 (Selection Pressure): 如果蒸馏返回 None，说明没通过幻觉校验
        if not distilled:
            print(f"[自然选择] 💥 记忆块 [{chunk.chunk_id[:8]}] 未通过适应度校验，已被永久抛弃 (Dropped)！")
            return
            
        chunk.distilled_version = distilled
        
        self.memory_database.append(chunk)
        # 更新记录，以便下一个块使用
        self.last_distilled_memory = distilled
        self.session_metrics["fidelity_scores"].append(distilled.fidelity_score)
        
        # 将记忆存入 SQLite 数据库
        if self.db_manager.save_memory_chunk(chunk):
            print("[系统] 记忆已持久化至 SQLite 数据库 (mled_memory.db)")
            # 【新增】更新检索器的 DAG 图
            self.retriever._rebuild_dag()
        else:
            print("[系统] ⚠️ 记忆持久化失败！")
        
        print("====== 蒸馏结果 ======")
        print(f"提取实体: {distilled.entities}")
        print(f"关键决策: {distilled.decisions}")
        print(f"事实与信息: {distilled.important_facts}")
        print(f"因果链接 (父节点): {distilled.parent_nodes}")
        
        # 【新增】显示元数据快照和推论
        if hasattr(distilled.structured_summary, 'get'):
            metadata = distilled.structured_summary.get("metadata", [])
            inferences = distilled.structured_summary.get("inferences", [])
            if metadata:
                print(f"元数据快照: {metadata}")
            if inferences:
                print(f"逻辑推演: {inferences}")
                
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
        """模拟后台运行的遗忘与状态更新周期 (淘汰机制)"""
        print("\n[系统后台] 开始执行动态遗忘与淘汰周期...")
        
        # 1. 执行动态遗忘
        for i, chunk in enumerate(self.memory_database):
            old_status = chunk.status
            # 调用 Rust 实现的高性能状态更新算法
            updated_chunk = self.dynamic_forgetting.update_memory_state(chunk)
            self.memory_database[i] = updated_chunk
            
            if old_status != updated_chunk.status:
                print(f"  -> 记忆块 [{updated_chunk.chunk_id[:8]}] 状态变更: {old_status} -> {updated_chunk.status}")
                
        # 2. 执行内存剪枝 (Pruning / Elimination)
        pruned_count = self.db_manager.prune_memories(fitness_threshold=0.3)
        if pruned_count > 0:
            print(f"  -> [淘汰机制] 发现了 {pruned_count} 条 fitness < 0.3 的低质量记忆，已将其彻底淘汰出记忆图谱！")
            self.retriever._rebuild_dag() # 剪枝后重建图谱
            
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
