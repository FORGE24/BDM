import json
from distiller import call_llm

def check_task_completion(user_input: str) -> bool:
    """
    意图识别：检测用户输入是否标志着一个任务的完成或话题的结束。
    如果是，我们应当立即触发一次记忆的强制分块（flush），保证这块任务记忆被完整封装。
    """
    prompt = f"""
    判断用户的以下发言是否具有“任务完成”、“话题终结”、“总结确认”的意图。
    
    特征提示：
    - "好的，就这样吧" -> 结束
    - "做完了，跑通了" -> 完成
    - "那我们下一个话题聊什么？" -> 话题切换（意味着上一个结束）
    - "继续"、"然后呢" -> 未结束
    - 普通的提问和聊天 -> 未结束
    
    [用户发言]: "{user_input}"
    
    请严格按照以下JSON格式输出：
    {{
        "is_completed": true 或 false
    }}
    """
    
    try:
        # 为了不拖慢每轮对话的速度，这里可以使用一个极小/极快的模型，但 MVP 我们先用相同的模型
        res_str = call_llm(prompt, system_msg="你是一个话题意图识别器。", response_format={"type": "json_object"})
        res_json = json.loads(res_str)
        return res_json.get("is_completed", False)
    except Exception as e:
        print(f"[意图识别错误] {e}")
        return False
