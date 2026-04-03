import numpy as np
from database import DatabaseManager
from memory import DistilledMemory
from sentence_transformers import SentenceTransformer

class MemoryRetriever:
    """混合检索器：关键字匹配 + 向量相似度 (MVP 版本)"""
    
    def __init__(self, db_manager: DatabaseManager, vector_weight: float = 0.7, keyword_weight: float = 0.3):
        self.db_manager = db_manager
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        # 初始化与 distiller 中相同的嵌入模型
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def _cosine_similarity(self, v1, v2):
        """计算余弦相似度"""
        if not v1 or not v2:
            return 0.0
        vec1 = np.array(v1)
        vec2 = np.array(v2)
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
    def retrieve_context(self, query: str, limit: int = 3) -> str:
        """
        检索相关的记忆块，组装为上下文字符串。
        结合了简单的子串匹配与向量相似度计算。
        """
        session = self.db_manager.get_session()
        try:
            # 1. 生成查询的向量
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # 获取所有存储的蒸馏记忆
            from database import DBDistilledMemory
            all_memories = session.query(DBDistilledMemory).all()
            
            scored_memories = []
            
            for memory in all_memories:
                keyword_score = 0
                
                # 关键字匹配得分
                fields_to_search = [
                    memory.entities,
                    memory.decisions,
                    memory.important_facts,
                    memory.preferences
                ]
                
                for field in fields_to_search:
                    if field:
                        for item in field:
                            if item in query or any(q in item for q in query.split() if len(q)>1):
                                keyword_score += 1
                
                # 向量相似度得分
                vector_score = 0.0
                if memory.embedding and len(memory.embedding) > 0:
                    vector_score = self._cosine_similarity(query_embedding, memory.embedding)
                    
                # 归一化关键词得分 (假设最高分为10，简单处理)
                normalized_keyword = min(1.0, keyword_score / 5.0)
                
                # 综合打分
                final_score = (self.vector_weight * vector_score) + (self.keyword_weight * normalized_keyword)
                
                if final_score > 0.1: # 设定一个最低阈值
                    scored_memories.append((final_score, memory))
                    
            # 排序并取前 N 个
            scored_memories.sort(key=lambda x: x[0], reverse=True)
            top_memories = [m for s, m in scored_memories[:limit]]
            
            if not top_memories:
                return "（无相关历史记忆）"
                
            context_parts = []
            for m in top_memories:
                part = f"- 曾提及实体: {m.entities}\n" \
                       f"  关键决策: {m.decisions}\n" \
                       f"  事实: {m.important_facts}\n" \
                       f"  偏好: {m.preferences}"
                context_parts.append(part)
                
            return "\n".join(context_parts)
            
        except Exception as e:
            print(f"[检索系统错误]: {e}")
            return "（检索失败）"
        finally:
            session.close()
