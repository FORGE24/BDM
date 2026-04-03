from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime, timezone
from bdm_rust import MemoryChunk, DistilledMemory

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
    embedding = Column(JSON, default=list)
    
    # 元数据
    compression_ratio = Column(Float, default=0.0)
    fidelity_score = Column(Float, default=1.0)
    generation_cost = Column(Integer, default=0)
    
    # DAG 因果关联 (Causal Linking)
    parent_nodes = Column(JSON, default=list)  # 前序节点 ID 列表
    child_nodes = Column(JSON, default=list)   # 后继节点 ID 列表
    
    # 逻辑推演标记
    is_inference = Column(Boolean, default=False)
    
    # 热度衰减权重
    heat_score = Column(Float, default=1.0)    # 初始热度分数
    last_accessed = Column(DateTime, default=datetime.now)  # 最后访问时间
    access_count = Column(Integer, default=0)   # 访问计数
    pruned = Column(Boolean, default=False)     # 是否被剪枝
    
    chunk = relationship("DBMemoryChunk", back_populates="distilled_memory")


class DBConsolidatedBlock(Base):
    __tablename__ = 'consolidated_blocks'

    consolidation_id = Column(String, primary_key=True)
    member_nodes = Column(JSON, default=list)  # 包含的原始节点 ID 列表
    meta_semantic = Column(JSON, default=list) # 高维元语义向量
    consolidation_score = Column(Float, default=1.0)
    timestamp = Column(Integer, default=0)
    collective_vitality = Column(Float, default=1.0) # Eros of Consolidation 集体生命值

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
                    embedding=dist.embedding,
                    compression_ratio=dist.compression_ratio,
                    fidelity_score=dist.fidelity_score,
                    generation_cost=dist.generation_cost,
                    parent_nodes=getattr(dist, 'parent_nodes', []),
                    heat_score=getattr(dist, 'heat_score', 1.0),
                    is_inference=getattr(dist, 'is_inference', False),
                    access_count=0
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
            
    def load_memory_chunks(self):
        """从数据库加载所有记忆块"""
        session = self.get_session()
        try:
            db_chunks = session.query(DBMemoryChunk).all()
            memory_chunks = []
            for db_chunk in db_chunks:
                # 创建MemoryChunk实例
                chunk = MemoryChunk(db_chunk.raw_text, db_chunk.tokens, db_chunk.semantic_boundary)
                
                # 设置其他属性
                chunk.chunk_id = db_chunk.chunk_id
                
                # 修复: 确保 timestamp 和 last_accessed 是带有时区信息的 datetime 对象
                chunk.timestamp = db_chunk.timestamp.replace(tzinfo=timezone.utc) if db_chunk.timestamp and db_chunk.timestamp.tzinfo is None else db_chunk.timestamp
                chunk.last_accessed = db_chunk.last_accessed.replace(tzinfo=timezone.utc) if db_chunk.last_accessed and db_chunk.last_accessed.tzinfo is None else db_chunk.last_accessed
                
                chunk.status = db_chunk.status
                chunk.access_count = db_chunk.access_count
                chunk.forgetting_rate = db_chunk.forgetting_rate
                chunk.recovery_potential = db_chunk.recovery_potential
                chunk.importance_score = db_chunk.importance_score
                chunk.related_chunks = db_chunk.related_chunks or []
                
                # 如果有蒸馏版本
                if db_chunk.distilled_memory:
                    db_dist = db_chunk.distilled_memory
                    dist = DistilledMemory(
                        source_chunk_id=db_dist.source_chunk_id,
                        structured_summary=db_dist.structured_summary,
                        entities=db_dist.entities,
                        decisions=db_dist.decisions,
                        actions=db_dist.actions,
                        constraints=db_dist.constraints,
                        preferences=db_dist.preferences,
                        code_snippets=db_dist.code_snippets,
                        important_facts=db_dist.important_facts,
                        compression_ratio=db_dist.compression_ratio,
                        fidelity_score=db_dist.fidelity_score,
                        generation_cost=db_dist.generation_cost,
                        embedding=db_dist.embedding,
                        parent_nodes=db_dist.parent_nodes,
                        heat_score=db_dist.heat_score,
                        is_inference=db_dist.is_inference
                    )
                    dist.memory_id = db_dist.memory_id
                    chunk.distilled_version = dist
                else:
                    chunk.distilled_version = None
                
                memory_chunks.append(chunk)
            
            return memory_chunks
        except Exception as e:
            print(f"数据库加载失败: {e}")
            return []
        finally:
            session.close()

    def get_all_nodes_with_embeddings(self):
        """获取所有带有嵌入向量的蒸馏记忆节点，用于局部巩固"""
        session = self.get_session()
        try:
            db_distilled = session.query(DBDistilledMemory).filter(DBDistilledMemory.embedding != None).all()
            nodes = []
            for dist in db_distilled:
                if dist.embedding:
                    nodes.append((dist.memory_id, dist.embedding))
            return nodes
        except Exception as e:
            print(f"获取节点向量失败: {e}")
            return []
        finally:
            session.close()
            
    def save_consolidated_block(self, block_info: dict):
        """保存合并生成的巩固块"""
        session = self.get_session()
        try:
            db_block = DBConsolidatedBlock(
                consolidation_id=block_info["consolidation_id"],
                member_nodes=block_info["member_nodes"],
                meta_semantic=block_info.get("meta_semantic", []), # 确保 advanced_features.py 中传递了这个字段
                consolidation_score=block_info.get("consolidation_score", 1.0),
                timestamp=block_info.get("timestamp", 0),
                collective_vitality=block_info.get("collective_vitality", 1.0)
            )
            session.add(db_block)
            session.commit()
            return True
        except Exception as e:
            print(f"保存巩固块失败: {e}")
            session.rollback()
            return False
        finally:
            session.close()
