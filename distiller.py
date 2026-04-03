import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI
from memory import MemoryChunk, DistilledMemory
from sentence_transformers import SentenceTransformer

# 抑制 SentenceTransformer 加载权重的警告
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

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

def call_llm(prompt: str, model: str = "deepseek-chat", response_format: dict = None, system_msg: str = "你是一个高度智能的系统结构化摘要引擎。") -> str:
    """调用 Deepseek API 生成回复"""
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
    }
    if response_format:
        kwargs["response_format"] = response_format
        
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content

def self_validate_summary(summary_json: dict, raw_text: str, parent_contexts: str = "") -> tuple[bool, str]:
    """自我校验蒸馏出的摘要是否准确反映了原始对话或继承自父节点"""
    
    inherited_prompt = ""
    if parent_contexts:
        inherited_prompt = f"""
        [父节点继承上下文 (Inherited Context)]：
        如果在摘要中出现了[原始对话]中没有，但存在于以下[父节点继承上下文]中的内容，
        这是【合法的上下文继承】，**绝对不要**将其判定为幻觉。
        
        {parent_contexts}
        """

    prompt = f"""
    请评估以下由AI生成的结构化摘要是否准确、完整地反映了对话的核心内容。
    
    [当前块原始对话]：
    {raw_text}
    
    {inherited_prompt}
    
    [生成的结构化摘要]：
    {json.dumps(summary_json, ensure_ascii=False, indent=2)}
    
    你需要判断：
    1. 是否存在严重的"幻觉"（无中生有，添加了【既不在当前块对话中，也不在父节点上下文中，且无法被合理推导】的实体、决策、事实等）？
    2. 是否遗漏了非常关键的决策或实体？
    3. 摘要中的所有内容是否都能在【当前块对话】或【父节点上下文】中找到依据，或者是由 AI 合理推导出的内容（Inference）？
    
    【严格要求】：
    - 摘要中提取的实体、决策和事实等，必须来源于用户的对话内容或故事设定。
    - **逻辑推论例外**：如果 `is_inference` 为 true，或者内容位于 `inferences` 字段中，这是 AI 的逻辑推演，属于合法内容，**不算作幻觉**。
    - **系统状态的快照**（如激活了哪个专家、系统处于什么模式）是 Memory DAG 的一部分，这些属于【状态元数据 (Metadata)】。
    - 如果摘要的 `metadata` 字段中包含了“MLED”、“DeepSeek”、“记忆挖掘专家已激活”等系统状态词汇，这是**合法**的，不算作幻觉。
    - 但如果把系统状态词汇错误地放入了 `entities`（实体）或 `important_facts`（事实）中，则判定为幻觉。
    - 允许继承父节点上下文中的信息，这不算幻觉。
    
    请严格按照以下JSON格式输出评估结果：
    {{
        "passed": true或false,
        "reason": "通过或不通过的原因，详细说明哪些内容是真正的幻觉"
    }}
    """
    
    try:
        eval_str = call_llm(prompt, response_format={"type": "json_object"}, system_msg="你是一个理性的质量检查员。你懂得区分什么是'上下文继承'，什么是'无中生有的幻觉'。")
        eval_json = json.loads(eval_str)
        return eval_json.get("passed", True), eval_json.get("reason", "未提供原因")
    except Exception as e:
        print(f"[自校验错误] {e}")
        return True, "校验过程出错，默认放行"

def self_distillation(chunk: MemoryChunk, all_memory_ids: list = None, parent_contexts: str = "") -> DistilledMemory:
    """
    结构化自蒸馏：从原始对话块提取结构化摘要 + 因果链接。
    引入血统因子 (parent_contexts)，允许上下文合法继承。
    """
    if all_memory_ids is None:
        all_memory_ids = []
        
    available_refs = json.dumps(all_memory_ids[-10:]) if all_memory_ids else "[]"
    
    inherited_prompt = ""
    if parent_contexts:
        inherited_prompt = f"""
        [你可以继承的父节点上下文]：
        {parent_contexts}
        注意：你可以将父节点上下文中的关键实体/事实与当前对话结合起来提取。
        """
        
    system_msg = """
    你是一个极其理性的信息提取引擎。
    【最高指令】：
    1. 你的任务是从[原始对话]中提取信息。如果存在[父节点上下文]，你可以将其与当前对话结合理解。
    2. 区分【用户故事内容】、【系统运行状态】和【逻辑推演】。
       - 将故事实体放在 `entities` 中。
       - 将系统运行状态（如：激活了某个专家、MLED系统状态）提取到 `metadata` 中。
       - **极其重要**：如果是AI（特别是逻辑推理专家）基于事实做出的**逻辑推断、假设或补充设定**，请将其放入 `inferences` 字段。
    3. 提取的信息必须能够追溯到当前对话或父节点上下文中，或者是基于这些上下文的合理推演。
    4. **环境分类 (context_tag)**：你必须将当前对话的语境分类为以下之一：
       - "math" (涉及计算、数字、物理等)
       - "story" (涉及剧情、人物、科幻设定等)
       - "reasoning" (涉及逻辑分析、为什么、假设推导等)
       - "general" (普通的日常对话，无明显特征)
    """
    
    prompt = f"""
    请对以下【唯一的原始对话片段】进行结构化提取：
    
    [原始对话片段开始]
    {chunk.raw_text}
    [原始对话片段结束]
    
    {inherited_prompt}
    
    可用的前序记忆块 ID: {available_refs}
    如果本记忆与之前的记忆有【明确的逻辑或因果依赖】，请选择最多 3 个父节点ID。如果没有，留空。
    
    请严格按照以下JSON格式输出（必须返回JSON格式）：
    {{
        "entities": ["实体1", ...],
        "decisions": ["决策1", ...],
        "actions": ["行动1", ...],
        "preferences": ["偏好1", ...],
        "important_facts": ["事实1", ...],
        "inferences": ["推论1", "推论2", ...],
        "metadata": ["系统状态快照1", "使用的专家模块", ...],
        "parent_nodes": ["前序记忆块ID1", ...],
        "context_tag": "math/story/reasoning/general"
    }}
    """
    
    max_retries = 2
    is_valid = False
    structured_summary = {}
    
    for attempt in range(max_retries + 1):
        try:
            summary_str = call_llm(prompt, response_format={"type": "json_object"}, system_msg=system_msg)
            structured_summary = json.loads(summary_str)
            
            is_valid, reason = self_validate_summary(structured_summary, chunk.raw_text, parent_contexts)
            if is_valid:
                break
                
            print(f"[蒸馏自校验拦截] 第 {attempt + 1} 次尝试失败: {reason}")
            if attempt == max_retries:
                print("🚨 [蒸馏自校验] 重试耗尽，启动【自然选择淘汰机制】(Selection Pressure) -> 直接丢弃该记忆！")
                return None  # 返回 None，意味着放弃该记忆的保存
            else:
                prompt += f"\n\n[上一次提取失败原因]：{reason}\n请严格修正，去除真正的幻觉，但保留合法的继承上下文！"
                
        except Exception as e:
            print(f"自蒸馏解析失败: {e}")
            return None
            
    # 生成向量嵌入 (Vector Embedding)
    # 将推论 inferences 也加入到向量化文本中
    embedding_text = f"实体: {' '.join(structured_summary.get('entities', []))} | 决策: {' '.join(structured_summary.get('decisions', []))} | 事实: {' '.join(structured_summary.get('important_facts', []))} | 推论: {' '.join(structured_summary.get('inferences', []))}"
    # 如果全为空，则不生成无意义的向量
    if len(embedding_text) < 20:
        embedding_vector = []
    else:
        embedding_vector = embedding_model.encode(embedding_text).tolist()
    
    distilled = DistilledMemory(
        source_chunk_id=chunk.chunk_id,
        structured_summary=structured_summary,
        entities=structured_summary.get("entities", []),
        decisions=structured_summary.get("decisions", []),
        actions=structured_summary.get("actions", []),
        preferences=structured_summary.get("preferences", []),
        code_snippets=[],
        important_facts=structured_summary.get("important_facts", []) + structured_summary.get("inferences", []), # 推论也作为事实存储，以便后续检索
        constraints=[],
        compression_ratio=0.3,
        fidelity_score=1.0 if is_valid else 0.1,
        generation_cost=0,
        embedding=embedding_vector,
        parent_nodes=structured_summary.get("parent_nodes", []),
        heat_score=1.0,
        context_tag=structured_summary.get("context_tag", "general"),
        fitness=0.8,        # 初始适应度
        success_rate=0.5,
        usage_count=0
    )
    
    return distilled
