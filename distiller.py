import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from memory import MemoryChunk, DistilledMemory
from sentence_transformers import SentenceTransformer

load_dotenv()

# 初始化本地的句子嵌入模型
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 使用 OpenAI 兼容的客户端调用 Deepseek API
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    # 提供一个默认的占位符，避免在未配置时直接崩溃，而是推迟到实际调用时处理
    api_key = "not_set"
    
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1"
)

def call_llm(prompt: str, model: str = "deepseek-chat", response_format: dict = {"type": "json_object"}, system_msg: str = "你是一个高度智能的系统结构化摘要引擎。") -> str:
    """调用 Deepseek API 生成回复"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        response_format=response_format
    )
    return response.choices[0].message.content

def self_validate_summary(summary_json: dict, raw_text: str) -> tuple[bool, str]:
    """自我校验蒸馏出的摘要是否准确反映了原始对话"""
    prompt = f"""
    请评估以下由AI生成的结构化摘要是否准确、完整地反映了原始对话的核心内容。
    
    [原始对话]：
    {raw_text}
    
    [生成的结构化摘要]：
    {json.dumps(summary_json, ensure_ascii=False, indent=2)}
    
    你需要判断：
    1. 是否存在严重的“幻觉”（无中生有）？
    2. 是否遗漏了非常关键的决策或实体？
    
    请严格按照以下JSON格式输出评估结果：
    {{
        "passed": true或false,
        "reason": "通过或不通过的原因"
    }}
    """
    
    try:
        eval_str = call_llm(prompt, system_msg="你是一个无情的质量检查员，专门检查内容摘要的准确性。")
        eval_json = json.loads(eval_str)
        return eval_json.get("passed", True), eval_json.get("reason", "未提供原因")
    except Exception as e:
        print(f"[自校验错误] {e}")
        return True, "校验过程出错，默认放行"

def lower_compression(current_level: str) -> str:
    """降级压缩要求，保留更多细节"""
    if current_level == "high":
        return "balanced"
    elif current_level == "balanced":
        return "low"
    return "low"

def self_distillation(chunk: MemoryChunk, previous_distilled: DistilledMemory = None, compression_level: str = "balanced", max_retries: int = 1) -> DistilledMemory:
    """
    结构化自蒸馏：从原始对话块提取结构化摘要。
    如果提供了 previous_distilled，则将其纳入上下文，使新记忆继承上一轮的50%精华。
    """
    compression_prompts = {
        "high": "生成极简摘要(50-100 tokens)，只保留核心决策",
        "balanced": "生成标准摘要(100-200 tokens)，保留关键信息",
        "low": "生成详细摘要(200-500 tokens)，保留多数细节"
    }
    
    prev_context = ""
    if previous_distilled:
        prev_context = f"""
    [上一轮对话的高浓缩记忆块] (请将以下重要信息以 50% 的压缩率融入到本次新蒸馏结果中，保持记忆的连贯性):
    - 实体: {previous_distilled.entities}
    - 决策: {previous_distilled.decisions}
    - 事实: {previous_distilled.important_facts}
    - 偏好: {previous_distilled.preferences}
        """
    
    prompt = f"""
    请对以下对话进行结构化摘要：
    {prev_context}
    
    [本次原始对话]：
    {chunk.raw_text}
    
    请严格按照以下JSON格式输出（必须返回JSON格式）：
    {{
        "entities": ["实体1", "实体2", ...],
        "decisions": ["决策1", "决策2", ...],
        "actions": ["行动1", "行动2", ...],
        "preferences": ["偏好1", "偏好2", ...],
        "code_snippets": ["代码片段1", "代码片段2", ...],
        "important_facts": ["事实1", "事实2", ...],
        "constraints": ["约束1", "约束2", ...]
    }}
    
    要求：{compression_prompts.get(compression_level, compression_prompts["balanced"])}
    """
    
    try:
        summary_str = call_llm(prompt)
        structured_summary = json.loads(summary_str)
        
        # 增加自我校验环节 (Self-Validation)
        fidelity_score = 1.0
        is_valid, reason = self_validate_summary(structured_summary, chunk.raw_text)
        if not is_valid:
            print(f"[蒸馏自校验失败] 原因: {reason}")
            if max_retries > 0:
                new_level = lower_compression(compression_level)
                print(f"[蒸馏自校验] 尝试降低压缩率({new_level})并重试...")
                # 惩罚一次重试带来的保真度降低
                retried_distilled = self_distillation(chunk, previous_distilled, new_level, max_retries - 1)
                retried_distilled.fidelity_score = 0.8
                return retried_distilled
            else:
                print("[蒸馏自校验] 重试次数耗尽，强行保存当前不完美结果。")
                fidelity_score = 0.5
        
        # 生成向量嵌入 (Vector Embedding)
        embedding_text = f"实体: {' '.join(structured_summary.get('entities', []))} | 决策: {' '.join(structured_summary.get('decisions', []))} | 事实: {' '.join(structured_summary.get('important_facts', []))}"
        embedding_vector = embedding_model.encode(embedding_text).tolist()
        
        # 构建 DistilledMemory
        distilled = DistilledMemory(
            source_chunk_id=chunk.chunk_id,
            structured_summary=structured_summary,
            entities=structured_summary.get("entities", []),
            decisions=structured_summary.get("decisions", []),
            actions=structured_summary.get("actions", []),
            preferences=structured_summary.get("preferences", []),
            code_snippets=structured_summary.get("code_snippets", []),
            important_facts=structured_summary.get("important_facts", []),
            constraints=structured_summary.get("constraints", []),
            compression_ratio=len(summary_str) / max(1, chunk.tokens),
            fidelity_score=fidelity_score,
            embedding=embedding_vector
        )
        return distilled
    except Exception as e:
        print(f"自蒸馏失败: {e}")
        # 降级或返回空
        return DistilledMemory(
            source_chunk_id=chunk.chunk_id,
            structured_summary={"error": str(e)},
            embedding=[]
        )
