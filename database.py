from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

Base = declarative_base()

class DBMemoryChunk(Base):
    __tablename__ = 'memory_chunks'

    chunk_id = Column(String, primary_key=True)
    raw_text = Column(Text, nullable=False)
    tokens = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    semantic_boundary = Column(Boolean, default=False)
    
    # 记忆状态
    status = Column(String, default="active")
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.now)
    
    # 遗忘/恢复参数
    forgetting_rate = Column(Float, default=0.01)
    recovery_potential = Column(Float, default=1.0)
    importance_score = Column(Float, default=1.0)
    
    # 关联
    related_chunks = Column(JSON, default=list) # 存储关联块的ID列表
    distilled_memory = relationship("DBDistilledMemory", back_populates="chunk", uselist=False)


class DBDistilledMemory(Base):
    __tablename__ = 'distilled_memories'

    memory_id = Column(String, primary_key=True)
    source_chunk_id = Column(String, ForeignKey('memory_chunks.chunk_id'))
    structured_summary = Column(JSON)
    
    # 结构化字段
    entities = Column(JSON, default=list)
    decisions = Column(JSON, default=list)
    actions = Column(JSON, default=list)
    constraints = Column(JSON, default=list)
    preferences = Column(JSON, default=list)
    code_snippets = Column(JSON, default=list)
    important_facts = Column(JSON, default=list)
    
    # 元数据
    compression_ratio = Column(Float, default=0.0)
    fidelity_score = Column(Float, default=1.0)
    generation_cost = Column(Integer, default=0)
    
    chunk = relationship("DBMemoryChunk", back_populates="distilled_memory")


class DatabaseManager:
    def __init__(self, db_path="sqlite:///mled_memory.db"):
        self.engine = create_engine(db_path, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
    def get_session(self):
        return self.Session()
        
    def save_memory_chunk(self, chunk):
        """将 Pydantic 模型 MemoryChunk 转换为 SQLAlchemy 模型并保存"""
        session = self.get_session()
        try:
            # 1. 构造数据库层的 Chunk
            db_chunk = DBMemoryChunk(
                chunk_id=chunk.chunk_id,
                raw_text=chunk.raw_text,
                tokens=chunk.tokens,
                timestamp=chunk.timestamp,
                semantic_boundary=chunk.semantic_boundary,
                status=chunk.status,
                access_count=chunk.access_count,
                last_accessed=chunk.last_accessed,
                forgetting_rate=chunk.forgetting_rate,
                recovery_potential=chunk.recovery_potential,
                importance_score=chunk.importance_score,
                related_chunks=chunk.related_chunks
            )
            session.add(db_chunk)
            
            # 2. 如果存在蒸馏版本，则一同保存
            if chunk.distilled_version:
                dist = chunk.distilled_version
                db_distilled = DBDistilledMemory(
                    memory_id=dist.memory_id,
                    source_chunk_id=chunk.chunk_id,
                    structured_summary=dist.structured_summary,
                    entities=dist.entities,
                    decisions=dist.decisions,
                    actions=dist.actions,
                    constraints=dist.constraints,
                    preferences=dist.preferences,
                    code_snippets=dist.code_snippets,
                    important_facts=dist.important_facts,
                    compression_ratio=dist.compression_ratio,
                    fidelity_score=dist.fidelity_score,
                    generation_cost=dist.generation_cost
                )
                session.add(db_distilled)
                
            session.commit()
            return True
        except Exception as e:
            print(f"数据库保存失败: {e}")
            session.rollback()
            return False
        finally:
            session.close()
