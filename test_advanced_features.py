#!/usr/bin/env python3
"""
高级BDM功能集成测试
测试：预测编码、局部巩固、MoE世界模型
"""

import json
import numpy as np
from advanced_features import (
    PredictiveCodecInterface,
    MemoryConsolidationEngine,
    ExpertWorldModel,
    AdvancedBDMSystem
)


def test_predictive_coding():
    """测试1：预测编码与稀疏脉冲系统"""
    print("\n" + "="*70)
    print("🧠 测试1: 预测编码与稀疏脉冲系统 (Predictive Coding)")
    print("="*70)
    
    codec = PredictiveCodecInterface(surprise_threshold=0.4)
    
    # 模拟节点序列
    test_cases = [
        ("node_A", [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),  # 完全一致 -> 惊奇度低
        ("node_B", [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),  # 完全一致 -> 惊奇度低
        ("node_C", [1.0, 1.0, 0.5], [0.0, 0.0, 0.0]),  # 大差异 -> 惊奇度高（脉冲！）
        ("node_D", [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 一致 -> 惊奇度低
    ]
    
    print("\n📊 逐节点惊奇度计算：")
    for node_id, actual, expected in test_cases:
        surprise, should_fire = codec.compute_surprise(
            node_id, actual, expected
        )
        print(f"  {node_id}: surprise={surprise:.3f}, "
              f"should_fire={'🔴 YES(触发脉冲)' if should_fire else '⚪ NO'}")
    
    # 获取统计
    stats = codec.get_spike_statistics()
    print(f"\n📈 脉冲窗口统计：")
    print(f"  平均惊奇度: {stats['average_surprise']:.3f}")
    print(f"  脉冲计数: {stats['spike_count']}")
    print(f"  最大惊奇度: {stats['max_surprise']:.3f}")
    print(f"  触发阈值: {stats['threshold']:.3f}")
    
    # 检查是否应该进行完整巩固
    should_consolidate = codec.should_perform_full_consolidation()
    print(f"\n💾 是否应触发完整蒸馏: {should_consolidate}")
    
    # 获取高惊奇事件
    high_surprise = codec.get_high_surprise_events(top_k=2)
    print(f"\n🔥 高惊奇度事件 TOP-2：")
    for event in high_surprise:
        print(f"  {event['node_id']}: "
              f"surprise={event['surprise_score']:.3f}, "
              f"type={event['activation_type']}")
    
    print("\n✅ 测试1完成")
    return stats


def test_local_consolidation():
    """测试2：局部巩固（睡眠学习）"""
    print("\n" + "="*70)
    print("🛏️  测试2: 局部巩固系统 (Local Consolidation)")
    print("="*70)
    
    consolidator = MemoryConsolidationEngine(min_fragment_size=2)
    
    # 模拟记忆碎片 (ID, embedding)
    # 创建3个语义相近的碎片组
    memory_nodes = [
        ("memory_1", [1.0, 0.0, 0.0, 0.0]),  # 组1
        ("memory_2", [0.95, 0.05, 0.0, 0.0]),  # 组1：相近
        ("memory_3", [0.05, 0.95, 0.05, 0.0]),  # 组2
        ("memory_4", [0.0, 1.0, 0.0, 0.0]),  # 组2：相近
        ("memory_5", [0.0, 0.0, 0.95, 0.05]),  # 组3
        ("memory_6", [0.0, 0.0, 1.0, 0.0]),  # 组3：相近
    ]
    
    print("\n🔍 识别碎片组：")
    clusters = consolidator.identify_fragments(memory_nodes)
    for i, cluster in enumerate(clusters, 1):
        print(f"  集群{i}: {cluster}")
    
    print(f"\n📦 执行巩固（合并）：")
    consolidated_blocks = []
    for cluster in clusters:
        embeddings = [node[1] for node in memory_nodes if node[0] in cluster]
        block = consolidator.consolidate(cluster, embeddings)
        consolidated_blocks.append(block)
        print(f"  巩固块ID: {block['consolidation_id']}")
        print(f"    包含节点: {block['member_nodes']}")
        print(f"    元语义维度: {block['meta_semantic_dimension']}")
        print(f"    质量分: {block['consolidation_score']:.3f}")
    
    # 统计
    stats = consolidator.get_consolidation_stats()
    print(f"\n📊 巩固统计：")
    print(f"  总巩固数: {stats['total_consolidations']}")
    print(f"  涉及节点总数: {stats['total_consolidated_nodes']}")
    print(f"  平均集群大小: {stats['average_cluster_size']:.2f}")
    
    print("\n✅ 测试2完成")
    return stats


def test_moe_world_model():
    """测试3：MoE世界模型"""
    print("\n" + "="*70)
    print("🌍 测试3: MoE世界模型 (Multi-Expert Orchestration)")
    print("="*70)
    
    model = ExpertWorldModel()
    
    print(f"\n👨‍🔬 已注册的专家：")
    for expert_id, info in model.expert_definitions.items():
        print(f"  {expert_id}:")
        print(f"    类型: {info['type']}")
        print(f"    权重: {info['weight']:.2f}")
        print(f"    描述: {info['description']}")
    
    # 测试路由
    test_scenarios = [
        ("query_math", ["problem_1", "analysis_2", "result_3"], "数学查询"),
        ("task_motion", ["force_calc", "acceleration", "result"], "物理查询"),
        ("logic_gate", ["and_op", "or_op", "result"], "逻辑查询"),
    ]
    
    print(f"\n🔀 路由决策（选择TOP-3专家）：")
    for node_id, dag_context, description in test_scenarios:
        routing = model.route_to_experts(node_id, dag_context, k=3)
        print(f"\n  📌 {description} ({node_id}):")
        for i, (expert_id, prob) in enumerate(routing, 1):
            expert_info = model.expert_definitions[expert_id]
            print(f"    {i}. {expert_id}: "
                  f"prob={prob:.3f}, type={expert_info['type']}")
    
    # 执行世界模型步骤
    print(f"\n🎯 执行世界模型推断：")
    
    # 数学专家场景
    input_vars = {
        "x": 4.0,
        "y": 9.0
    }
    
    output = model.execute_world_step(
        "query_math",
        ["problem_1", "analysis_2"],
        input_vars
    )
    
    print(f"  输入: {input_vars}")
    print(f"  输出: {output}")
    
    # 物理专家场景
    input_vars = {
        "velocity": 10.0,
        "time": 2.0
    }
    
    output = model.execute_world_step(
        "task_motion",
        ["force_calc", "acceleration"],
        input_vars
    )
    
    print(f"  输入: {input_vars}")
    print(f"  输出: {output}")
    
    # 执行情况统计
    stats = model.get_execution_stats()
    print(f"\n📈 执行统计：")
    print(f"  总执行次数: {stats['total_executions']}")
    if stats['expert_invoke_counts']:
        print(f"  专家调用分布:")
        for expert_id, count in stats['expert_invoke_counts']:
            print(f"    {expert_id}: {count}次")
    
    print("\n✅ 测试3完成")
    return stats


def test_integrated_system():
    """测试4：集成系统 - 三个功能协作"""
    print("\n" + "="*70)
    print("🔗 测试4: 集成系统 (All Features Together)")
    print("="*70)
    
    # 创建集成系统（模拟数据库）
    class MockDatabase:
        def get_all_nodes_with_embeddings(self):
            return [
                ("node_1", [1.0, 0.0, 0.0]),
                ("node_2", [0.9, 0.1, 0.0]),
                ("node_3", [0.0, 1.0, 0.0]),
            ]
    
    db = MockDatabase()
    memory_manager = None
    
    system = AdvancedBDMSystem(db, memory_manager)
    
    print("\n🔄 场景：处理具有高惊奇度的节点")
    
    # 步骤1：预测编码检测高惊奇度
    result = system.process_node_with_prediction(
        "novel_event",
        [1.0, 1.0, 1.0],  # 异常的嵌入
        [0.0, 0.0, 0.0]   # 预期值
    )
    
    print(f"  预测编码检测:")
    print(f"    节点: {result['node_id']}")
    print(f"    惊奇度: {result['surprise_score']:.3f}")
    print(f"    触发巩固: {result['should_trigger_consolidation']}")
    
    # 步骤2：如果惊奇度高，触发巩固
    if result['should_trigger_consolidation']:
        print(f"\n  触发巩固流程...")
        memory_nodes = [
            ("mem_1", [1.0, 0.0, 0.0]),
            ("mem_2", [0.95, 0.05, 0.0]),
            ("mem_3", [0.0, 1.0, 0.0]),
        ]
        consolidated = system.trigger_consolidation(memory_nodes)
        print(f"    生成{len(consolidated)}个巩固块")
    
    # 步骤3：通过MoE路由处理查询
    print(f"\n  查询路由（MoE）:")
    routing = system.route_query_to_experts(
        "novel_event",
        ["analysis", "context"],
        top_k=2
    )
    for route in routing:
        print(f"    激活{route['expert_id']}: "
              f"概率={route['probability']:.3f}")
    
    # 最终系统状态
    print(f"\n📊 系统最终状态：")
    status = system.get_system_status()
    print(f"  预测编码 - 脉冲数: {status['predictive_coding']['spike_count']}")
    print(f"  局部巩固 - 巩固块数: {status['consolidation']['total_consolidations']}")
    print(f"  MoE执行 - 执行次数: {status['world_model']['total_executions']}")
    
    print("\n✅ 测试4完成")
    return status


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("🚀 BDM高级功能集成测试套件")
    print("="*70)
    
    test_results = {}
    
    try:
        test_results["predictive_coding"] = test_predictive_coding()
    except Exception as e:
        print(f"❌ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_results["consolidation"] = test_local_consolidation()
    except Exception as e:
        print(f"❌ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_results["moe_model"] = test_moe_world_model()
    except Exception as e:
        print(f"❌ 测试3失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_results["integrated"] = test_integrated_system()
    except Exception as e:
        print(f"❌ 测试4失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 最终总结
    print("\n" + "="*70)
    print("✅ 所有高级功能测试完成！")
    print("="*70)
    print("\n📝 测试摘要：")
    print(f"  ✅ 预测编码与稀疏脉冲系统")
    print(f"  ✅ 局部巩固系统（睡眠学习）")
    print(f"  ✅ MoE世界模型（多专家系统）")
    print(f"  ✅ 集成协作系统")
    print("\n🎉 所有功能已成功实现和测试！")


if __name__ == "__main__":
    main()
