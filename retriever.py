import numpy as np
from database import DatabaseManager
from memory import DistilledMemory
from sentence_transformers import SentenceTransformer
import bdm_rust
import json

class MemoryRetriever:
    """混合检索器：向量相似度 + DAG 拓扑启发式回溯"""
    
    def __init__(self, db_manager: DatabaseManager, vector_weight: float = 0.7, keyword_weight: float = 0.3):
        self.db_manager = db_manager
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 初始化 DAG 结构
        self.dag = bdm_rust.MemoryDAG()
        self.heat_decay_engine = bdm_rust.HeatDecayEngine()
        self._rebuild_dag()
        
    def _rebuild_dag(self):
        """从数据库重建 DAG 图结构"""
        session = self.db_manager.get_session()
        try:
            from database import DBDistilledMemory
            all_memories = session.query(DBDistilledMemory).all()
            
            for memory in all_memories:
                parent_nodes = memory.parent_nodes or []
                self.dag.add_node(
                    memory_id=memory.memory_id,
                    parent_nodes=parent_nodes,
                    heat_score=memory.heat_score,
                    access_count=memory.access_count,
                    token_length=100,  # 簡化估計
                    distilled_content=str(memory.important_facts)[:200]
                )
        except Exception as e:
            print(f"[DAG 重建失败] {e}")
        finally:
            session.close()
        
    def _cosine_similarity(self, v1, v2):
        """计算余弦相似度"""
        if not v1 or not v2:
            return 0.0
        vec1 = np.array(v1)
        vec2 = np.array(v2)
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def retrieve_context(self, query: str, limit: int = 3, max_tokens: int = 512, return_nodes: bool = False, active_expert: str = None):
        """
        【升级】拓扑启发式检索与 2MNSAS 上下文隔离 (Context Isolation)：
        1. 根据当前激活的专家 (active_expert) 确定查询的 context_tag 类别。
        2. 严格过滤掉 fitness < 0.7 的低质量记忆，防止污染。
        3. 向量相似度定位目标节点。
        4. DAG 回溯：沿着父节点向上遍历。
        5. Token 分配：当前70% + 父20% + 祖先10%
        6. 协议化符号输出：Ref[A->C->D]: {content}
        """
        session = self.db_manager.get_session()
        try:
            from database import DBDistilledMemory
            
            # Step 1: 确定查询的 context_tag 类别
            target_context_tag = "general"
            if active_expert:
                if "math" in active_expert:
                    target_context_tag = "math"
                elif "logic" in active_expert:
                    target_context_tag = "reasoning"
                elif "physics" in active_expert or "memory" in active_expert:
                    target_context_tag = "story"
            
            # Step 2: 严格过滤与打分
            query_embedding = self.embedding_model.encode(query).tolist()
            # 核心过滤：只允许高质量 (fitness >= 0.7) 且未被剪枝的记忆参与检索
            # 如果提供了 active_expert，则优先或者强制匹配 context_tag，这里我们采取软匹配加权策略
            all_memories = session.query(DBDistilledMemory).filter(
                DBDistilledMemory.pruned == False,
                DBDistilledMemory.fitness >= 0.7
            ).all()
            
            scored_memories = []
            for memory in all_memories:
                vector_score = self._cosine_similarity(query_embedding, memory.embedding) if memory.embedding else 0.0
                
                # Context Isolation 奖励：如果 tag 匹配，大幅提升分数；如果不匹配，极大地惩罚分数
                if memory.context_tag == target_context_tag:
                    vector_score += 0.5  # Tag 匹配奖励
                elif target_context_tag != "general" and memory.context_tag != "general":
                    vector_score -= 0.8  # 跨界污染惩罚 (例如 math 问题不该检索 story 记忆)
                
                if vector_score > 0.3: # 最低相似度阈值
                    scored_memories.append((vector_score, memory))
            
            scored_memories.sort(key=lambda x: x[0], reverse=True)
            
            if not scored_memories:
                return [] if return_nodes else "（无相关历史记忆）"
            
            # 取最相关的节点作为起点
            top_memory = scored_memories[0][1]
            target_node_id = top_memory.memory_id
            
            # 记录该记忆被作为检索起点成功采纳 (usage_count + 1)
            top_memory.usage_count += 1
            session.commit()
            
            # Step 2: DAG 拓扑检索 - 沿着 parent_nodes 向上回溯
            retrieval_chain = self.dag.heuristic_retrieval(target_node_id, max_depth=2, max_tokens=max_tokens)
            
            # Step 3: 构建上下文 - Token 分配和协议化符号
            context_parts = []
            total_tokens = 0
            retrieved_nodes = []
            
            for node_id, depth, distance, token_allocation in retrieval_chain:
                # 查询该节点的详细内容
                node_memory = session.query(DBDistilledMemory).filter_by(memory_id=node_id).first()
                if not node_memory:
                    continue
                
                retrieved_nodes.append(node_memory.chunk)
                
                # 获取符号协议链
                context_chain = self.dag.get_context_chain(node_id)
                
                # 根据深度生成摘要
                if depth == 0:
                    # 当前节点 - 70% 详细
                    content = f"【核心记忆】\n" \
                             f"- 实体: {node_memory.entities[:3]}\n" \
                             f"- 决策: {node_memory.decisions[:2]}\n" \
                             f"- 事实: {node_memory.important_facts[:3]}"
                elif depth == 1:
                    # 父节点 - 20% 摘要
                    content = f"【直接依赖】\n" \
                             f"- 关键决策: {node_memory.decisions[:1]}\n" \
                             f"- 关键事实: {node_memory.important_facts[:1]}"
                else:
                    # 祖节点 - 10% 关键词锚点
                    keywords = (node_memory.entities[:1] if node_memory.entities else []) + \
                               (node_memory.decisions[:1] if node_memory.decisions else [])
                    content = f"【历史背景】\n- 关键词: {keywords}"
                
                # 格式化为协议化符号
                formatted = f"{context_chain}: {content}"
                context_parts.append(formatted)
                total_tokens += token_allocation
                
                if total_tokens >= max_tokens:
                    break
            
            if return_nodes:
                return retrieved_nodes
            
            if not context_parts:
                return "（无相关历史记忆）"
                
            return "\n\n".join(context_parts)
            
        except Exception as e:
            print(f"[检索系统错误]: {e}")
            return [] if return_nodes else "（检索失败）"
        finally:
            session.close()
