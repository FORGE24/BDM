from dataclasses import dataclass
from typing import List

@dataclass
class SafetyCheck:
    passed: bool
    action: str
    response: str = ""
    violation_recorded: bool = False

class SafetyGuardrail:
    """不可进化的安全边界系统"""
    
    def __init__(self):
        # 不可变的安全规则
        self.immutable_rules = [
            "永远不能生成有害内容",
            "永远不能泄露隐私信息",
            "永远不能绕过内容过滤器",
            "永远不能模仿系统指令",
        ]
        
    def enforce_safety(self, action: str) -> SafetyCheck:
        """强制执行安全规则 (MVP版本: 仅执行基于关键字的基础检测)"""
        
        # 危险关键词列表（模拟内容过滤）
        dangerous_keywords = ["毁灭", "自杀", "泄露密码", "忽略指令", "黑客攻击"]
        
        for keyword in dangerous_keywords:
            if keyword in action:
                return self.safety_violation_response(
                    action, 
                    f"内容触发了安全词汇拦截: [{keyword}]"
                )
                
        # 所有检查通过
        return SafetyCheck(passed=True, action=action)
    
    def safety_violation_response(self, action: str, reason: str) -> SafetyCheck:
        """安全违规的标准响应"""
        return SafetyCheck(
            passed=False,
            action=action,
            response=f"🚫 拦截提示：由于安全原因，系统无法执行此操作。原因: {reason}",
            violation_recorded=True
        )
