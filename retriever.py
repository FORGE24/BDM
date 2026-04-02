from database import DatabaseManager
from memory import DistilledMemory

class MemoryRetriever:
    """简单的基于关键字的混合检索器 (MVP 版本)"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    def retrieve_context(self, query: str, limit: int = 3) -> str:
        """
        检索相关的记忆块，组装为上下文字符串。
        MVP版通过简单的子串匹配来实现。
        """
        session = self.db_manager.get_session()
        try:
            # 获取所有存储的蒸馏记忆
            from database import DBDistilledMemory
            all_memories = session.query(DBDistilledMemory).all()
            
            scored_memories = []
            
            for memory in all_memories:
                score = 0
                
                # 在各种结构化字段中匹配
                fields_to_search = [
                    memory.entities,
                    memory.decisions,
                    memory.important_facts,
                    memory.preferences
                ]
                
                for field in fields_to_search:
                    if field:
                        for item in field:
                            # 如果用户问题中的词在记忆项中，或者反之，加分
                            if item in query or any(q in item for q in query.split() if len(q)>1):
                                score += 1
                                
                if score > 0:
                    scored_memories.append((score, memory))
                    
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
